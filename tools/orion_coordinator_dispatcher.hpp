// Sender-side Orion coordinator selection. C++17, libcurl and json-c.
// Use one instance per collection per sender process; share it across HTTP lanes.
#pragma once
#include <curl/curl.h>
#include <json-c/json.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace orion {
// A reused HTTP connection can close after the server accepted the query. libcurl
// can then replay a POST internally. Stop that replay before sending bytes: a
// rejection of the second copy cannot prove the first copy never started.
class SingleSubmission {
  public:
    void begin(CURL *http) {
        attempted_ = false;
        if (curl_easy_setopt(http, CURLOPT_PREREQFUNCTION, before_send) != CURLE_OK ||
            curl_easy_setopt(http, CURLOPT_PREREQDATA, this) != CURLE_OK) {
            throw std::runtime_error("libcurl 7.80 or later is required for single submission");
        }
    }

  private:
    bool attempted_ = false;
    static int before_send(void *context, char *, char *, int, int) {
        auto *submission = static_cast<SingleSubmission *>(context);
        return std::exchange(submission->attempted_, true) ? CURL_PREREQFUNC_ABORT
                                                           : CURL_PREREQFUNC_OK;
    }
};

class CoordinatorDispatcher {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::milliseconds;
    struct Block {
        std::string id;
        unsigned next, count;
        Clock::time_point deadline;
    };
    struct Peer {
        std::string collection_url;
        std::deque<Block> blocks;
        double weight = 0, deficit = 0;
        Clock::time_point fresh_until{}, refresh_at{}, earliest_refill{};

        unsigned available(Clock::time_point now) {
            blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
                                        [now](const Block &b) {
                                            return b.next == b.count || now >= b.deadline;
                                        }),
                         blocks.end());
            unsigned total = 0;
            for (const auto &b : blocks)
                total += b.count - b.next;
            return total;
        }
    };

  public:
    struct Destination {
        std::string url, lease;
        unsigned slot;
        size_t peer;
        std::string wrap(const std::string &search_json) const {
            return "{\"permit\":{\"lease\":\"" + lease + "\",\"slot\":" + std::to_string(slot) +
                   "},\"search\":" + search_json + "}";
        }
    };

    // Give the concrete collection URLs of the nodes. Auth and source address,
    // if used, should also be set on the application's data-plane HTTP client.
    explicit CoordinatorDispatcher(const std::vector<std::string> &urls,
                                   std::string source_address = {}, std::string api_key = {})
        : source_(std::move(source_address)), key_(std::move(api_key)) {
        if (urls.empty() || std::set<std::string>(urls.begin(), urls.end()).size() != urls.size())
            throw std::invalid_argument("distinct coordinator URLs are required");
        for (const auto &url : urls)
            peers_.push_back(Peer{url, {}});
        try {
            for (size_t i = 0; i < peers_.size(); ++i)
                threads_.emplace_back([this, i] { poll(i); });
        } catch (...) {
            stop();
            throw;
        }
    }
    CoordinatorDispatcher(const CoordinatorDispatcher &) = delete;
    CoordinatorDispatcher &operator=(const CoordinatorDispatcher &) = delete;
    ~CoordinatorDispatcher() { stop(); }

    // Call only when a send slot is available. An empty result leaves the query
    // in the sender's unassigned queue. No address is fixed ahead of the send.
    std::optional<Destination> try_pick() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = Clock::now();
        if (now - demand_ > Ms(100))
            changed_.notify_all();
        demand_ = now;
        Peer *best = nullptr;
        double sum = 0;
        for (auto &p : peers_) {
            if (now >= p.fresh_until || p.weight <= 0 || !p.available(now)) {
                p.deficit = 0;
                continue;
            }
            sum += p.weight;
            p.deficit += p.weight;
            if (!best || p.deficit > best->deficit)
                best = &p;
        }
        if (!best) {
            unassigned_ = true;
            return {};
        }
        unassigned_ = false;
        best->deficit -= sum;
        auto &block = best->blocks.front();
        Destination result{best->collection_url + "/orion/coordinator/search", block.id,
                           block.next++, static_cast<size_t>(best - peers_.data())};
        if (block.next == block.count)
            changed_.notify_all();
        return result;
    }

    // Wait outside the search service. A total query deadline belongs to the caller.
    void wait_until(Clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiters_;
        changed_.notify_all();
        changed_.wait_until(lock, deadline, [&] {
            auto now = Clock::now();
            for (auto &p : peers_)
                if (p.weight > 0 && now < p.fresh_until && p.available(now))
                    return true;
            return stopped_;
        });
        --waiters_;
    }

    // Only the explicit 412 with x-orion-admission: not-started permits reselection.
    // A connection error or query timeout after submission has ambiguous ownership.
    void rejected_before_start(const Destination &dest) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &p = peers_.at(dest.peer);
        p.blocks.erase(std::remove_if(p.blocks.begin(), p.blocks.end(),
                                      [&](const Block &b) { return b.id == dest.lease; }),
                       p.blocks.end());
        changed_.notify_all();
    }
    uint64_t feedback_requests() const { return probes_.load(); }

  private:
    static constexpr unsigned batch = 32; // transport amortization; the node may grant fewer
    std::vector<Peer> peers_;
    std::string source_, key_, identity_;
    uint64_t generation_ = 0;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::thread> threads_;
    std::atomic<uint64_t> probes_{0};
    bool stopped_ = false;
    size_t waiters_ = 0;
    bool unassigned_ = false;
    Clock::time_point demand_{};
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        changed_.notify_all();
        for (auto &t : threads_)
            if (t.joinable())
                t.join();
    }
    static size_t append(char *buf, size_t size, size_t n, void *dst) {
        auto &out = *static_cast<std::string *>(dst);
        if (size * n > 16384 - out.size())
            return 0;
        out.append(buf, size * n);
        return size * n;
    }
    static json_object *field(json_object *obj, const char *name, json_type type) {
        json_object *value = nullptr;
        if (!json_object_object_get_ex(obj, name, &value) || !json_object_is_type(value, type))
            throw std::runtime_error(std::string("invalid permit reply: ") + name);
        return value;
    }
    void poll(size_t i) noexcept {
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                                 curl_easy_cleanup);
        if (!curl)
            return;
        auto *list = curl_slist_append(nullptr, "Content-Type: application/json");
        if (!key_.empty())
            list = curl_slist_append(list, ("api-key: " + key_).c_str());
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(list,
                                                                            curl_slist_free_all);
        auto *c = curl.get();
        const auto url = peers_[i].collection_url + "/orion/coordinator/permits";
        curl_easy_setopt(c, CURLOPT_URL, url.c_str());
        curl_easy_setopt(c, CURLOPT_POST, 1L);
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, list);
        curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(c, CURLOPT_NOPROXY, "*");
        curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, 250L);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, append);
        if (!source_.empty())
            curl_easy_setopt(c, CURLOPT_INTERFACE, source_.c_str());
        while (true) {
            unsigned want = 0;
            bool waiting = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                auto &p = peers_[i];
                while (!stopped_) {
                    auto now = Clock::now();
                    if (!waiters_ && now - demand_ > Ms(100)) {
                        changed_.wait(lock);
                        continue;
                    }
                    unsigned left = p.available(now);
                    if (now >= p.refresh_at || (left <= batch / 4 && now >= p.earliest_refill)) {
                        want = left < batch ? batch - left : 0;
                        waiting = unassigned_ || waiters_ > 0;
                        break;
                    }
                    changed_.wait_until(lock, left <= batch / 4
                                                  ? std::min(p.earliest_refill, p.refresh_at)
                                                  : p.refresh_at);
                }
                if (stopped_)
                    return;
            }
            const auto sent = Clock::now();
            std::string body = "{\"count\":" + std::to_string(want) +
                               ",\"waiting\":" + (waiting ? "true}" : "false}"),
                        reply;
            try {
                curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
                curl_easy_setopt(c, CURLOPT_WRITEDATA, &reply);
                ++probes_;
                const auto ec = curl_easy_perform(c);
                long http = 0;
                curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http);
                if (ec != CURLE_OK || http != 200)
                    throw std::runtime_error("permit endpoint unavailable");
                std::unique_ptr<json_object, decltype(&json_object_put)> root(
                    json_tokener_parse(reply.c_str()), json_object_put);
                if (!root)
                    throw std::runtime_error("invalid permit response");
                auto *v = field(root.get(), "result", json_type_object);
                auto generation = json_object_get_uint64(field(v, "generation", json_type_int));
                auto identity =
                    std::to_string(generation) + "/" +
                    json_object_get_string(field(v, "artifact_sha256", json_type_string)) + "/" +
                    json_object_get_string(field(v, "policy_sha256", json_type_string));
                auto count = json_object_get_uint64(field(v, "count", json_type_int));
                auto ttl = Ms(json_object_get_int64(field(v, "ttl_ms", json_type_int)));
                auto weight = json_object_get_double(field(v, "weight", json_type_double));
                if (json_object_get_uint64(field(v, "limit", json_type_int)) == 0)
                    weight = 0;
                if (count > want || ttl.count() <= 0 || ttl.count() > 1000 ||
                    !std::isfinite(weight) || weight < 0)
                    throw std::runtime_error("invalid scheduler grant");
                std::string lease;
                if (count)
                    lease = json_object_get_string(field(v, "lease", json_type_string));
                if (count && (lease.size() != 36 ||
                              lease.find_first_not_of("0123456789abcdef-") != std::string::npos))
                    throw std::runtime_error("invalid grant ID");
                const auto now = Clock::now();
                // Use our own monotonic request start and leave an observed RTT for sending.
                const auto deadline = sent + ttl - (now - sent);
                std::lock_guard<std::mutex> lock(mutex_);
                auto &p = peers_[i];
                if (identity_.empty() || generation > generation_) {
                    generation_ = generation;
                    identity_ = identity;
                    for (auto &old : peers_) {
                        old.blocks.clear();
                        old.weight = 0;
                        old.deficit = 0;
                    }
                }
                if (identity != identity_)
                    throw std::runtime_error("incompatible coordinator identity");
                p.available(now);
                p.weight = weight;
                p.fresh_until = deadline;
                if (!weight || deadline <= now)
                    p.blocks.clear();
                else if (count)
                    p.blocks.push_back({lease, 0, static_cast<unsigned>(count), deadline});
                p.refresh_at = now + std::max(Ms(1), ttl / 2);
                p.earliest_refill = count ? now + std::max(Ms(1), ttl / 16) : p.refresh_at;
                changed_.notify_all();
            } catch (const std::exception &) {
                std::lock_guard<std::mutex> lock(mutex_);
                auto &p = peers_[i];
                p.blocks.clear();
                p.weight = 0;
                p.refresh_at = p.earliest_refill = Clock::now() + Ms(25);
            }
        }
    }
};
} // namespace orion
