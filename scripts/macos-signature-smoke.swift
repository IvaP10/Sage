// Read-only native checks for Command Line Tools installations without XCTest.
// Compiled with production sources and protobuf types; no alternative verifier.
import AppKit
import Foundation

private struct CheckFailure: Error { let message: String }
private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw CheckFailure(message: message) }
}
private func rejects(_ action: () throws -> Void) throws {
    do { try action() } catch { return }
    throw CheckFailure(message: "Unsafe input was accepted")
}

@main
struct SignatureSmoke {
    @MainActor
    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data("Signature smoke check failed: \(error)\n".utf8))
            exit(1)
        }
    }
    @MainActor
    static func run() async throws {
        let table = Data([0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 1,
                          1, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0, 0])
        try require(MachOSlices.offsets(header: table, fileSize: 96) == [64], "Valid architecture table was rejected")
        try rejects { _ = try MachOSlices.offsets(header: table.prefix(12), fileSize: 96) }
        try rejects { _ = try MachOSlices.offsets(header: table, fileSize: 95) }
        var tooMany = table; tooMany[7] = 33
        try rejects { _ = try MachOSlices.offsets(header: tooMany, fileSize: 96) }
        var overlapping = table; overlapping[7] = 2; overlapping.append(table.dropFirst(8))
        try rejects { _ = try MachOSlices.offsets(header: overlapping, fileSize: 96) }
        print("PASS: bounded architecture table accepts valid slices and rejects malformed inputs")
        let target = try SignedApplicationIdentity.inspect(URL(fileURLWithPath: "/System/Applications/Calculator.app"))
        try SignedApplicationIdentity.validate(target)
        for field in 0..<3 {
            var changed = target
            switch field {
            case 0: changed.codeDigest = String(repeating: "0", count: target.codeDigest.count)
            case 1: changed.signer = "unrelated-team"
            default: changed.bundlePath = "/System/Applications/Calendar.app"
            }
            try rejects { try SignedApplicationIdentity.validate(changed) }
        }
        print("PASS: signed system bundle accepted; changed path, signer and digest rejected")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let contents = root.appendingPathComponent("Untrusted.app/Contents")
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        let plist = try PropertyListSerialization.data(fromPropertyList: ["CFBundleIdentifier": "com.example.Untrusted", "CFBundlePackageType": "APPL"], format: .xml, options: 0)
        try plist.write(to: contents.appendingPathComponent("Info.plist"))
        try rejects { _ = try SignedApplicationIdentity.inspect(contents.deletingLastPathComponent()) }
        print("PASS: unsigned fixture bundle rejected")
        if let finder = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.finder").first, let url = finder.bundleURL {
            let identity = try SignedApplicationIdentity.inspect(url)
            try SignedApplicationIdentity.validateProcess(finder.processIdentifier, target: identity)
            try rejects { try SignedApplicationIdentity.validateProcess(0, target: identity) }
            print("PASS: existing Finder process matched to signed bundle; invalid PID rejected")
        } else { print("SKIP: no graphical Finder session; process signature check not exercised") }
        let preparationAdapter = PlatformAdapter()
        var preparation = Sage_Ipc_V2_AdapterRequest()
        preparation.requestID = UUID().uuidString
        preparation.operation = "application_identity"
        preparation.json = "{\"application\":\"Calculator\"}"
        preparation.expiresAtUnixMs = Int64(Date().timeIntervalSince1970 * 1000) + 30_000
        let runningBefore = Set(NSRunningApplication.runningApplications(withBundleIdentifier: target.identifier).map(\.processIdentifier))
        let started = DispatchTime.now().uptimeNanoseconds
        let located = await preparationAdapter.handle(preparation)
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        try require(located.success && located.applicationTarget == target, "Native preparation did not resolve the signed Calculator bundle: \(located.error)")
        try require(Set(NSRunningApplication.runningApplications(withBundleIdentifier: target.identifier).map(\.processIdentifier)) == runningBefore, "Preparation must not launch the app")
        let cancelled = Task { @MainActor in await preparationAdapter.handle(preparation) }
        cancelled.cancel()
        let cancelledResult = await cancelled.value
        try require(!cancelledResult.success, "Cancelled preparation must not produce a ready identity")
        print("PASS: actual native identity preparation and pre-admission cancellation; preparation_ns=\(elapsed)")
        var request = Sage_Ipc_V2_AdapterRequest()
        request.requestID = UUID().uuidString
        request.operation = "execute"
        request.json = "{\"action\":{\"type\":\"open_application\",\"application\":\"com.apple.calculator\"}}"
        request.expiresAtUnixMs = Int64(Date().timeIntervalSince1970 * 1000) + 10_000
        let result = await PlatformAdapter().handle(request)
        try require(!result.success && result.error.contains("Typed application grant"), "Untyped execution must be rejected")
        var grant = Sage_Ipc_V2_ExecutionGrant()
        grant.grantID = UUID().uuidString; grant.runID = UUID().uuidString
        grant.actionID = UUID().uuidString; grant.workerSession = UUID().uuidString
        grant.policyVersion = 2; grant.domain = "native"; grant.actionDigest = String(repeating: "0", count: 64)
        grant.expiresAtUnixMs = request.expiresAtUnixMs
        grant.application = "com.apple.calculator"; grant.operations = ["read"]
        request.grant = grant
        var identity = Sage_Ipc_V2_ApplicationTarget(); identity.identifier = "com.apple.calculator"
        request.applicationTarget = identity
        let wrongOperations = await PlatformAdapter().handle(request)
        try require(!wrongOperations.success && wrongOperations.error.contains("Typed application grant"), "Read authority must not permit an application launch")
        print("PASS: missing typed identity and incorrect operation authority rejected before launch")
        print("No application was launched or activated. These checks do not qualify the complete approval/launch flow.")
    }
}
