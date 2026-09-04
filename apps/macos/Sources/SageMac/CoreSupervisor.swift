import Foundation

@MainActor
final class CoreSupervisor {
    private var process: Process?

    func startIfNeeded(secret: Data) throws {
        guard process == nil else { return }
        let executable = try locateCore()
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--bootstrap-stdin"]
        let bootstrap = Pipe()
        process.standardInput = bootstrap
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        let encoded = Data((secret.base64EncodedString() + "\n").utf8)
        try bootstrap.fileHandleForWriting.write(contentsOf: encoded)
        try bootstrap.fileHandleForWriting.close()
        self.process = process
    }

    func detach() {
        // The core is intentionally daemon-like. Closing or restarting the UI must
        // not terminate active tasks. The OS/app lifecycle owns final shutdown.
        process = nil
    }

    private func locateCore() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["SAGE_CORE_EXECUTABLE"] {
            let url = URL(fileURLWithPath: override)
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw SageClientError.connectionFailed("SAGE_CORE_EXECUTABLE is not executable")
            }
            return url
        }
        let bundled = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers/sage-core", isDirectory: false)
        guard FileManager.default.isExecutableFile(atPath: bundled.path) else {
            throw SageClientError.connectionFailed(
                "sage-core is missing from Contents/Helpers; set SAGE_CORE_EXECUTABLE for source development"
            )
        }
        return bundled
    }
}
