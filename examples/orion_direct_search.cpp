// Send one Qdrant SearchRequest per stdin line to a worker-first coordinator.
#include "../tools/orion_coordinator_dispatcher.hpp"

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <map>
#include <poll.h>
#include <unistd.h>
#include <string_view>

namespace {
using Clock = std::chrono::steady_clock;

struct Reply {
    std::string body;
    bool not_started = false;
};

size_t receive(char *data, size_t size, size_t count, void *userdata) {
    static_cast<Reply *>(userdata)->body.append(data, size * count);
    return size * count;
}

size_t read_header(char *data, size_t size, size_t count, void *userdata) {
    std::string_view line(data, size * count);
    auto colon = line.find(':');
    if (colon != line.npos) {
        std::string name(line.substr(0, colon));
        for (char &c : name)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        auto value = line.substr(colon + 1);
        auto first = value.find_first_not_of(" \t");
        auto last = value.find_last_not_of(" \t\r\n");
        if (name == "x-orion-admission" && first != value.npos && last != value.npos &&
            value.substr(first, last - first + 1) == "not-started") {
            static_cast<Reply *>(userdata)->not_started = true;
        }
    }
    return size * count;
}

// Read at most one lookahead query when the window is full. poll() also keeps
// outstanding HTTP requests moving while the input pipe is open but idle.
class Input {
    std::string buffer_;
    size_t offset_ = 0;
    bool eof_ = false;

  public:
    bool finished() const { return eof_ && offset_ == buffer_.size(); }

    std::optional<std::string> next() {
        for (;;) {
            auto end = buffer_.find('\n', offset_);
            if (end != std::string::npos || (eof_ && offset_ < buffer_.size())) {
                if (end == std::string::npos)
                    end = buffer_.size();
                auto line = buffer_.substr(offset_, end - offset_);
                offset_ = end + (end < buffer_.size());
                if (!line.empty())
                    return line;
                continue;
            }
            if (eof_)
                return {};
            pollfd input{STDIN_FILENO, POLLIN, 0};
            int ready = poll(&input, 1, 0);
            if (ready < 0 && errno == EINTR)
                continue;
            if (ready < 0)
                throw std::runtime_error("cannot poll query input");
            if (!ready)
                return {};
            char chunk[16384];
            auto n = read(STDIN_FILENO, chunk, sizeof(chunk));
            if (n < 0 && errno == EINTR)
                continue;
            if (n < 0)
                throw std::runtime_error("cannot read query input");
            buffer_.erase(0, offset_);
            offset_ = 0;
            if (n == 0)
                eof_ = true;
            else
                buffer_.append(chunk, static_cast<size_t>(n));
        }
    }
};

struct Query {
    const size_t id;
    const std::string json;
    const Clock::time_point deadline = Clock::now() + std::chrono::seconds(60);
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> http{curl_easy_init(), curl_easy_cleanup};
    std::optional<orion::CoordinatorDispatcher::Destination> destination;
    orion::SingleSubmission submission;
    std::string body;
    Reply reply;
    CURLM *attached = nullptr;

    Query(size_t id, std::string json, curl_slist *headers, const char *source)
        : id(id), json(std::move(json)) {
        if (!http)
            throw std::runtime_error("cannot initialize the HTTP client");
        curl_easy_setopt(http.get(), CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(http.get(), CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(http.get(), CURLOPT_NOPROXY, "*");
        curl_easy_setopt(http.get(), CURLOPT_WRITEFUNCTION, receive);
        curl_easy_setopt(http.get(), CURLOPT_HEADERFUNCTION, read_header);
        curl_easy_setopt(http.get(), CURLOPT_WRITEDATA, &reply);
        curl_easy_setopt(http.get(), CURLOPT_HEADERDATA, &reply);
        if (source)
            curl_easy_setopt(http.get(), CURLOPT_INTERFACE, source);
    }

    void send(CURLM *multi, orion::CoordinatorDispatcher::Destination dest) {
        destination = std::move(dest);
        reply = {};
        body = destination->wrap(json);
        auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        if (remaining.count() <= 0)
            throw std::runtime_error("coordinator admission deadline elapsed");
        curl_easy_setopt(http.get(), CURLOPT_URL, destination->url.c_str());
        curl_easy_setopt(http.get(), CURLOPT_TIMEOUT_MS, static_cast<long>(remaining.count()));
        curl_easy_setopt(http.get(), CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(http.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                         static_cast<curl_off_t>(body.size()));
        submission.begin(http.get());
        auto code = curl_multi_add_handle(multi, http.get());
        if (code != CURLM_OK)
            throw std::runtime_error(curl_multi_strerror(code));
        attached = multi;
    }

    void detach() {
        if (attached)
            curl_multi_remove_handle(std::exchange(attached, nullptr), http.get());
    }
    ~Query() { detach(); }
};

struct Result {
    bool ok;
    std::string text;
};

int run(const std::vector<std::string> &urls) {
    const char *key = std::getenv("QDRANT_API_KEY");
    const char *source = std::getenv("ORION_SOURCE_ADDRESS");
    orion::CoordinatorDispatcher dispatcher(urls, source ? source : "", key ? key : "");
    std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)> multi(curl_multi_init(),
                                                                curl_multi_cleanup);
    if (!multi)
        throw std::runtime_error("cannot initialize HTTP transfers");
    auto *headers = curl_slist_append(nullptr, "Content-Type: application/json");
    if (key)
        headers = curl_slist_append(headers, (std::string("api-key: ") + key).c_str());
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> owned_headers(headers,
                                                                              curl_slist_free_all);
    Input input;
    std::deque<std::unique_ptr<Query>> pending;
    std::map<CURL *, std::unique_ptr<Query>> active;
    std::map<size_t, Result> completed;
    size_t next_id = 0, next_output = 0, peak_active = 1;
    bool failed = false;
    for (;;) {
        bool submitted = false;
        // Pump the transport between small batches of immediately available permits.
        for (unsigned i = 0; i < 32; ++i) {
            if (pending.empty()) {
                if (failed || completed.size() >= peak_active)
                    break;
                auto json = input.next();
                if (!json)
                    break;
                pending.push_back(
                    std::make_unique<Query>(next_id++, std::move(*json), headers, source));
            }
            if (Clock::now() >= pending.front()->deadline) {
                completed.emplace(pending.front()->id,
                                  Result{false, "coordinator admission deadline elapsed"});
                pending.pop_front();
                failed = true;
                continue;
            }
            auto dest = dispatcher.try_pick();
            if (!dest)
                break;
            auto job = std::move(pending.front());
            pending.pop_front();
            job->send(multi.get(), std::move(*dest));
            CURL *handle = job->http.get();
            active.emplace(handle, std::move(job));
            submitted = true;
        }
        peak_active = std::max(peak_active, active.size());
        int running = 0;
        auto status = curl_multi_perform(multi.get(), &running);
        if (status != CURLM_OK)
            throw std::runtime_error(curl_multi_strerror(status));
        int left;
        bool received = false;
        while (auto *msg = curl_multi_info_read(multi.get(), &left)) {
            if (msg->msg != CURLMSG_DONE)
                continue;
            const auto transport = msg->data.result;
            auto job = std::move(active.extract(msg->easy_handle).mapped());
            long http_status = 0;
            curl_easy_getinfo(job->http.get(), CURLINFO_RESPONSE_CODE, &http_status);
            job->detach();
            received = true;
            if (transport == CURLE_OK && http_status == 412 && job->reply.not_started) {
                dispatcher.rejected_before_start(*job->destination);
                pending.push_back(std::move(job));
                continue;
            }
            const bool ok = transport == CURLE_OK && http_status == 200;
            if (!ok)
                failed = true;
            std::string message;
            if (transport != CURLE_OK)
                message = curl_easy_strerror(transport);
            else if (ok)
                message = std::move(job->reply.body);
            else
                message = "HTTP " + std::to_string(http_status) + " " + job->reply.body;
            completed.emplace(job->id, Result{ok, std::move(message)});
        }
        bool emitted = false;
        for (;;) {
            auto entry = completed.find(next_output);
            if (entry == completed.end())
                break;
            if (!entry->second.ok)
                throw std::runtime_error(entry->second.text);
            std::cout << entry->second.text << '\n';
            completed.erase(entry);
            ++next_output;
            emitted = true;
        }
        if (emitted)
            std::cout.flush();
        if (active.empty() && pending.empty() && input.finished())
            return 0;
        if (received || submitted)
            continue;
        if (active.empty() && !pending.empty()) {
            dispatcher.wait_until(pending.front()->deadline);
            continue;
        }
        const bool read_input =
            !failed && !input.finished() && pending.empty() && completed.size() < peak_active;
        curl_waitfd fd{STDIN_FILENO, CURL_WAIT_POLLIN, 0};
        // While waiting for admission, also service active sockets and pick up new grants.
        status = curl_multi_poll(multi.get(), read_input ? &fd : nullptr, read_input ? 1 : 0,
                                 pending.empty() ? 1000 : 5, nullptr);
        if (status != CURLM_OK)
            throw std::runtime_error(curl_multi_strerror(status));
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: orion_direct_search COLLECTION_URL [COLLECTION_URL ...]\n";
        return 2;
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
        return 1;
    int status = 1;
    try {
        status = run({argv + 1, argv + argc});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
    }
    curl_global_cleanup();
    return status;
}
