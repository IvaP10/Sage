import AppKit
import XCTest
@testable import SageMac

final class SignedApplicationIdentityTests: XCTestCase {
    func testArchitectureTableRejectsTruncationOverlapAndUnboundedCounts() throws {
        let valid = Data([0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 1,
                          1, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 32, 0, 0, 0, 0])
        XCTAssertEqual(try MachOSlices.offsets(header: valid, fileSize: 96), [64])
        XCTAssertThrowsError(try MachOSlices.offsets(header: valid.prefix(12), fileSize: 96))
        XCTAssertThrowsError(try MachOSlices.offsets(header: valid, fileSize: 95))
        var tooMany = valid; tooMany[7] = 33
        XCTAssertThrowsError(try MachOSlices.offsets(header: tooMany, fileSize: 96))
        var overlapping = valid; overlapping[7] = 2; overlapping.append(valid.dropFirst(8))
        XCTAssertThrowsError(try MachOSlices.offsets(header: overlapping, fileSize: 96))
    }
    func testSystemBundleIdentityRejectsChangedCodeSignerAndPath() throws {
        let target = try SignedApplicationIdentity.inspect(URL(fileURLWithPath: "/System/Applications/Calculator.app"))
        XCTAssertEqual(target.platform, "macos")
        XCTAssertFalse(target.codeDigest.isEmpty)
        try SignedApplicationIdentity.validate(target)
        var changed = target
        changed.codeDigest = String(repeating: "0", count: target.codeDigest.count)
        XCTAssertThrowsError(try SignedApplicationIdentity.validate(changed))
        changed = target
        changed.signer = "unrelated-team"
        XCTAssertThrowsError(try SignedApplicationIdentity.validate(changed))
        changed = target
        changed.bundlePath = "/System/Applications/Calendar.app"
        XCTAssertThrowsError(try SignedApplicationIdentity.validate(changed))
    }

    func testUnsignedBundleCannotBecomeAnApplicationTarget() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let contents = root.appendingPathComponent("Untrusted.app/Contents")
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        let plist = try PropertyListSerialization.data(fromPropertyList: ["CFBundleIdentifier": "com.example.Untrusted", "CFBundlePackageType": "APPL"], format: .xml, options: 0)
        try plist.write(to: contents.appendingPathComponent("Info.plist"))
        XCTAssertThrowsError(try SignedApplicationIdentity.inspect(contents.deletingLastPathComponent()))
    }

    func testExistingFinderProcessMatchesItsSignedBundle() throws {
        guard let finder = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.finder").first,
              let url = finder.bundleURL else { throw XCTSkip("No graphical Finder session is running") }
        let target = try SignedApplicationIdentity.inspect(url)
        try SignedApplicationIdentity.validateProcess(finder.processIdentifier, target: target)
        XCTAssertThrowsError(try SignedApplicationIdentity.validateProcess(0, target: target))
    }

    @MainActor
    func testAdapterRejectsExecutionWithoutTypedIdentityBeforeDispatch() async throws {
        var request = Sage_Ipc_V2_AdapterRequest()
        request.requestID = UUID().uuidString
        request.operation = "execute"
        request.json = "{\"action\":{\"type\":\"open_application\",\"application\":\"com.apple.calculator\"}}"
        request.expiresAtUnixMs = Int64(Date().timeIntervalSince1970 * 1000) + 10_000
        let result = await PlatformAdapter().handle(request)
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("Typed application grant"))
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
        XCTAssertFalse(wrongOperations.success)
        XCTAssertTrue(wrongOperations.error.contains("Typed application grant"))
    }
}
