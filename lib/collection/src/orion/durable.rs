//! Portable no-clobber directory publication. NFS may not support renameat2
//! NOREPLACE on directories; publish the validated commit record last instead.
use std::fs::{self, File};
use std::path::Path;

pub fn publish_directory(source: &Path, target: &Path, commit_file: &str) -> std::io::Result<()> {
    // Atomically reserve the target name; even an existing empty directory is rejected.
    fs::create_dir(target)?;
    fn payloads(source: &Path, target: &Path, skip: Option<&str>) -> std::io::Result<()> {
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let name = entry.file_name();
            if skip.is_some_and(|skip| name == skip) {
                continue;
            }
            let kind = entry.file_type()?;
            if kind.is_dir() {
                fs::create_dir(target.join(&name))?;
                payloads(&entry.path(), &target.join(&name), None)?;
                fs::remove_dir(entry.path())?;
            } else if kind.is_file() {
                atomicwrites::move_atomic(&entry.path(), &target.join(&name))?;
            } else {
                return Err(std::io::Error::other(
                    "non-regular file in Orion publication",
                ));
            }
        }
        File::open(target)?.sync_all()?;
        Ok(())
    }
    payloads(source, target, Some(commit_file))?;
    atomicwrites::move_atomic(&source.join(commit_file), &target.join(commit_file))?;
    File::open(target)?.sync_all()?;
    if let Some(parent) = target.parent() {
        File::open(parent)?.sync_all()?;
    }
    fs::remove_dir(source)?;
    Ok(())
}
