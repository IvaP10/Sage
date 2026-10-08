import Foundation
import Security
import CryptoKit

/// Signature inspection has no authority to launch an application. Callers run
/// the expensive validation away from the main actor, then recheck the grant.
enum SignedApplicationIdentity {
    static func inspect(_ url: URL) throws -> Sage_Ipc_V2_ApplicationTarget {
        let canonical = url.resolvingSymlinksInPath().standardizedFileURL
        guard canonical.pathExtension == "app", let bundle = Bundle(url: canonical),
              let identifier = bundle.bundleIdentifier else { throw failure("Target is not an installed application bundle") }
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(canonical as CFURL, [], &code) == errSecSuccess, let code else {
            throw failure("Application signature is unavailable")
        }
        let flags = SecCSFlags(rawValue: kSecCSCheckAllArchitectures | kSecCSCheckNestedCode | kSecCSStrictValidate)
        guard SecStaticCodeCheckValidity(code, flags, try requirement("anchor apple generic")) == errSecSuccess else {
            throw failure("Application requires a valid Apple-issued code signature with intact resources")
        }
        let info = try signingInformation(code)
        guard info[kSecCodeInfoIdentifier] as? String == identifier,
              let executable = info[kSecCodeInfoMainExecutable] as? URL,
              executable.resolvingSymlinksInPath().path.hasPrefix(canonical.path + "/") else {
            throw failure("Application bundle and code identity do not match")
        }
        let signer = try signer(info, staticCode: code)
        let file = try FileHandle(forReadingFrom: executable)
        defer { try? file.close() }
        let fileSize = try file.seekToEnd()
        try file.seek(toOffset: 0)
        let offsets = try MachOSlices.offsets(header: file.read(upToCount: 4096) ?? Data(), fileSize: fileSize)
        var digests = Set<String>()
        for offset in offsets {
            var selected: SecStaticCode? = code
            if offset != 0 {
                guard SecStaticCodeCreateWithPathAndAttributes(canonical as CFURL, [],
                    [kSecCodeAttributeUniversalFileOffset: offset] as CFDictionary, &selected) == errSecSuccess else {
                    throw failure("Application architecture could not be inspected")
                }
            }
            guard let selected,
                  SecStaticCodeCheckValidity(selected, SecCSFlags(rawValue: kSecCSStrictValidate), try requirement("anchor apple generic")) == errSecSuccess else {
                throw failure("Application architecture failed signature validation")
            }
            let slice = try signingInformation(selected)
            guard slice[kSecCodeInfoIdentifier] as? String == identifier,
                  try self.signer(slice, staticCode: selected) == signer,
                  let digest = slice[kSecCodeInfoUnique] as? Data, !digest.isEmpty else {
                throw failure("Application architectures do not share the same signed identity")
            }
            digests.insert(digest.map { String(format: "%02x", $0) }.joined())
        }
        var target = Sage_Ipc_V2_ApplicationTarget()
        target.platform = "macos"
        target.bundlePath = canonical.path
        target.identifier = identifier
        target.codeDigests = digests.sorted()
        target.codeDigest = SHA256.hash(data: Data(target.codeDigests.joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
        target.signer = signer
        return target
    }

    static func validate(_ target: Sage_Ipc_V2_ApplicationTarget) throws {
        guard target.platform == "macos", target.bundlePath.hasPrefix("/"),
              !target.identifier.isEmpty, !target.codeDigest.isEmpty, !target.signer.isEmpty,
              try inspect(URL(fileURLWithPath: target.bundlePath)) == target else {
            throw failure("Application changed after preparation; a new approval is required")
        }
    }

    static func validateProcess(_ pid: Int32, target: Sage_Ipc_V2_ApplicationTarget) throws {
        guard pid > 0 else { throw failure("Application has no running process") }
        try validate(target)
        var code: SecCode?
        guard SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributePid: pid] as CFDictionary, [], &code) == errSecSuccess,
              let code, SecCodeCheckValidity(code, [], try requirement("anchor apple generic")) == errSecSuccess else {
            throw failure("Running application failed dynamic signature validation")
        }
        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode else {
            throw failure("Running process code identity is unavailable")
        }
        let info = try signingInformation(staticCode)
        guard let actualExecutable = info[kSecCodeInfoMainExecutable] as? URL,
              actualExecutable.resolvingSymlinksInPath() == Bundle(url: URL(fileURLWithPath: target.bundlePath))?.executableURL?.resolvingSymlinksInPath(),
              let digest = info[kSecCodeInfoUnique] as? Data,
              target.codeDigests.contains(digest.map({ String(format: "%02x", $0) }).joined()),
              info[kSecCodeInfoIdentifier] as? String == target.identifier,
              try signer(info, staticCode: staticCode) == target.signer else {
            throw failure("Running process is not the approved signed application")
        }
    }

    private static func signer(_ info: [CFString: Any], staticCode: SecStaticCode) throws -> String {
        if let team = info[kSecCodeInfoTeamIdentifier] as? String, !team.isEmpty { return team }
        guard SecStaticCodeCheckValidity(staticCode, [], try requirement("anchor apple")) == errSecSuccess else {
            throw failure("Application has no verifiable signing team")
        }
        return "APPLE"
    }
    private static func signingInformation(_ code: SecStaticCode) throws -> [CFString: Any] {
        var dictionary: CFDictionary?
        guard SecCodeCopySigningInformation(code, SecCSFlags(rawValue: kSecCSSigningInformation), &dictionary) == errSecSuccess,
              let result = dictionary as? [CFString: Any] else { throw failure("Code signature information is incomplete") }
        return result
    }
    private static func requirement(_ text: String) throws -> SecRequirement {
        var result: SecRequirement?
        guard SecRequirementCreateWithString(text as CFString, [], &result) == errSecSuccess,
              let result else { throw failure("Code signing requirement is unavailable") }
        return result
    }
    private static func failure(_ text: String) -> Error { SageClientError.protocolError(text) }
}
