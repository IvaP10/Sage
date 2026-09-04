import Darwin
import Foundation

/// Stores only the local IPC authentication key. This is transport material,
/// not a provider credential, so opening Sage never triggers Keychain UI.
/// Provider credentials remain Keychain-backed and are loaded on demand.
struct IPCSecretStore {
    private let secretLength = 32

    func loadOrCreateSecret() throws -> Data {
        let directory = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("Sage", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: directory.path
        )
        let keyURL = directory.appendingPathComponent("ipc-auth.key", isDirectory: false)
        do {
            return try loadSecret(at: keyURL)
        } catch let error as NSError where error.domain == NSPOSIXErrorDomain && error.code == Int(ENOENT) {
            return try createSecret(at: keyURL)
        }
    }

    private func loadSecret(at url: URL) throws -> Data {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { throw posixError() }
        defer { Darwin.close(descriptor) }

        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0 else { throw posixError() }
        let kind = metadata.st_mode & mode_t(S_IFMT)
        guard kind == mode_t(S_IFREG), metadata.st_uid == geteuid() else {
            throw SageClientError.authenticationFailed(
                "The local IPC key is not an owner-controlled regular file"
            )
        }
        guard metadata.st_mode & mode_t(S_IRWXG | S_IRWXO) == 0 else {
            throw SageClientError.authenticationFailed(
                "The local IPC key permissions must be 0600"
            )
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        let secret = try handle.readToEnd() ?? Data()
        guard secret.count == secretLength else {
            throw SageClientError.authenticationFailed(
                "The local IPC key has an invalid length"
            )
        }
        return secret
    }

    private func createSecret(at url: URL) throws -> Data {
        var secret = Data(count: secretLength)
        let randomStatus = secret.withUnsafeMutableBytes { buffer in
            SecRandomCopyBytes(kSecRandomDefault, secretLength, buffer.baseAddress!)
        }
        guard randomStatus == errSecSuccess else {
            throw SageClientError.authenticationFailed(
                "Secure random generation failed: \(randomStatus)"
            )
        }

        let descriptor = Darwin.open(
            url.path,
            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,
            mode_t(S_IRUSR | S_IWUSR)
        )
        if descriptor < 0 {
            if errno == EEXIST { return try loadSecret(at: url) }
            throw posixError()
        }
        defer { Darwin.close(descriptor) }
        guard fchmod(descriptor, mode_t(S_IRUSR | S_IWUSR)) == 0 else {
            throw posixError()
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        try handle.write(contentsOf: secret)
        try handle.synchronize()
        return secret
    }

    private func posixError() -> NSError {
        NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
}
