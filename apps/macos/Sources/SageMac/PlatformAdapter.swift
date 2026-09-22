import AppKit
import ApplicationServices
import Foundation

/// Native OS access lives in the client; only an authenticated core request can
/// reach this adapter. It never plans, approves, or grants filesystem access.
@MainActor
final class PlatformAdapter {
    private var previousApplication: NSRunningApplication?
    private var activationObserver: NSObjectProtocol?
    private var usedGrants: [String: Date] = [:]

    init() {
        previousApplication = NSWorkspace.shared.frontmostApplication
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
                  app.processIdentifier != ProcessInfo.processInfo.processIdentifier else { return }
            MainActor.assumeIsolated { self?.previousApplication = app }
        }
    }

    func handle(_ request: Sage_Ipc_V2_AdapterRequest) async -> Sage_Ipc_V2_AdapterResult {
        var result = Sage_Ipc_V2_AdapterResult()
        result.requestID = request.requestID
        do {
            guard request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else { throw failure("Adapter request expired") }
            var payload = try JSONSerialization.jsonObject(with: Data(request.json.utf8)) as? [String: Any] ?? [:]
            if request.operation == "execute" {
                let grant = request.grant
                guard request.hasGrant, grant.policyVersion == 2, grant.domain == "native",
                      grant.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000),
                      case .application(let identifier) = grant.resource else { throw failure("Missing typed application grant") }
                payload["capability"] = ["id": grant.grantID, "task_id": grant.runID, "action_id": grant.actionID,
                    "domain": grant.domain, "policy_version": grant.policyVersion, "remaining_uses": 0, "revoked": false,
                    "expires_at": ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: Double(grant.expiresAtUnixMs) / 1000)),
                    "resource": ["kind": "application", "identifier": identifier]] as [String: Any]
            }
            let data: [String: Any]
            switch request.operation {
            case "context": data = context()
            case "observe": data = try observe(payload)
            case "execute": data = try await execute(payload)
            default: throw failure("Unknown adapter operation")
            }
            result.json = String(decoding: try JSONSerialization.data(withJSONObject: data), as: UTF8.self)
            result.success = true
        } catch { result.error = error.localizedDescription }
        return result
    }

    private func context() -> [String: Any] {
        let front = NSWorkspace.shared.frontmostApplication
        let app = front?.processIdentifier == ProcessInfo.processInfo.processIdentifier ? previousApplication : front
        guard let app else { return ["available": false] }
        var result: [String: Any] = ["active_application": app.bundleIdentifier ?? "", "application_name": app.localizedName ?? "", "accessibility_available": AXIsProcessTrusted()]
        result["screen_state"] = NSScreen.screens.map { ["width": $0.frame.width, "height": $0.frame.height, "scale": $0.backingScaleFactor] }
        guard AXIsProcessTrusted() else { return result }
        let root = AXUIElementCreateApplication(app.processIdentifier)
        if let window = element(root, kAXFocusedWindowAttribute) {
            result["active_window"] = string(window, kAXTitleAttribute)
            result["current_resource"] = string(window, kAXDocumentAttribute)
            result["accessibility_tree"] = descendants(window, limit: 80).map { item in
                ["role": string(item, kAXRoleAttribute), "label": label(item), "automation_id": string(item, kAXIdentifierAttribute)]
            }
        }
        if let focused = element(root, kAXFocusedUIElementAttribute), !secure(focused) {
            result["selected_text"] = String(string(focused, kAXSelectedTextAttribute).prefix(4000))
            result["selection"] = ["role": string(focused, kAXRoleAttribute), "label": label(focused)]
        }
        return result
    }

    private func observe(_ payload: [String: Any]) throws -> [String: Any] {
        let condition = payload["condition"] as? [String: Any] ?? [:]
        switch condition["kind"] as? String {
        case "application_running":
            let identifier = condition["application"] as? String ?? ""
            return ["running": running(identifier) != nil]
        case "element_present":
            try requireAccessibility()
            let identifier = payload["application"] as? String ?? previousApplication?.bundleIdentifier ?? ""
            guard let app = running(identifier) else { return ["present": false] }
            let matches = try matching(app, condition["selector"] as? [String: Any] ?? [:])
            return ["present": matches.count == 1]
        default: throw failure("Observation is not supported by the native adapter")
        }
    }

    private func execute(_ payload: [String: Any]) async throws -> [String: Any] {
        let action = payload["action"] as? [String: Any] ?? [:]
        let grant = payload["capability"] as? [String: Any] ?? [:]
        let resource = grant["resource"] as? [String: Any] ?? [:]
        let identifier = action["application"] as? String ?? ""
        let grantID = grant["id"] as? String ?? ""
        guard !grantID.isEmpty, grant["domain"] as? String == "native",
              grant["remaining_uses"] as? Int == 0, grant["revoked"] as? Bool == false,
              resource["kind"] as? String == "application", resource["identifier"] as? String == identifier,
              !identifier.isEmpty, identifier != Bundle.main.bundleIdentifier else { throw failure("Native capability does not match this action") }
        let formatter = ISO8601DateFormatter(); formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let expiry = grant["expires_at"] as? String ?? ""
        guard let expires = formatter.date(from: expiry) ?? ISO8601DateFormatter().date(from: expiry), expires > Date() else { throw failure("Native capability expired") }
        guard action["type"] as? String == "open_application" else { throw failure("This native operation has no enabled feature contract") }
        usedGrants = usedGrants.filter { $0.value > Date() }
        guard usedGrants[grantID] == nil else { throw failure("Native capability already used") }
        usedGrants[grantID] = expires
        if action["type"] as? String == "open_application" {
            if let app = running(identifier) { app.activate(); return [:] }
            let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: identifier)
                ?? ["/Applications", "/System/Applications"].map { URL(fileURLWithPath: $0).appendingPathComponent(identifier + ".app") }.first { FileManager.default.fileExists(atPath: $0.path) && !$0.path.contains("..") }
            guard let url else { throw failure("Application could not be resolved to an installed bundle") }
            _ = try await NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
            return [:]
        }
        guard let app = running(identifier) else { throw failure("Target application is no longer running") }
        if action["type"] as? String == "close_application" {
            guard app.terminate() else { throw failure("Application declined to close") }; return [:]
        }
        try requireAccessibility()
        switch action["type"] as? String {
        case "click_element":
            let target = try unique(app, action["selector"] as? [String: Any] ?? [:])
            guard !secure(target), AXUIElementPerformAction(target, kAXPressAction as CFString) == .success else { throw failure("Element does not support a safe accessibility press") }
        case "type_text":
            guard let selector = action["selector"] as? [String: Any] else { throw failure("Typing requires an explicit semantic selector") }
            let target = try unique(app, selector)
            guard !secure(target), action["sensitive"] as? Bool != true else { throw failure("Use the native secure field for sensitive input") }
            var settable = DarwinBoolean(false)
            guard AXUIElementIsAttributeSettable(target, kAXValueAttribute as CFString, &settable) == .success, settable.boolValue else { throw failure("Target is not an editable accessibility element") }
            let text = action["text"] as? String ?? ""
            guard AXUIElementSetAttributeValue(target, kAXValueAttribute as CFString, text as CFString) == .success else { throw failure("Accessibility edit failed") }
        case "press_shortcut":
            let keys = (action["keys"] as? [String] ?? []).map { $0.lowercased() }.sorted().joined(separator: "+")
            let shortcuts: [String: CGKeyCode] = ["command+s": 1, "command+f": 3, "command+a": 0, "command+c": 8, "command+z": 6]
            guard let code = shortcuts[keys] else { throw failure("Shortcut is not in the native allowlist") }
            app.activate()
            try await Task.sleep(for: .milliseconds(150))
            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier else { throw failure("Target application lost focus") }
            for down in [true, false] {
                guard let event = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: down) else { throw failure("Keyboard event unavailable") }
                event.flags = .maskCommand; event.postToPid(app.processIdentifier)
            }
        default: throw failure("No structured adapter is installed for this application action")
        }
        return [:]
    }

    private func running(_ identifier: String) -> NSRunningApplication? {
        let matches = NSWorkspace.shared.runningApplications.filter { $0.bundleIdentifier == identifier || $0.localizedName?.caseInsensitiveCompare(identifier) == .orderedSame }
        return matches.count == 1 ? matches[0] : nil
    }

    private func requireAccessibility() throws {
        guard AXIsProcessTrusted() else { throw failure("Enable Accessibility for Sage in System Settings to use this action") }
    }

    private func matching(_ app: NSRunningApplication, _ selector: [String: Any]) throws -> [AXUIElement] {
        let role = selector["role"] as? String
        let name = selector["label"] as? String
        let id = selector["automation_id"] as? String
        guard [role, name, id].contains(where: { $0?.isEmpty == false }) else { throw failure("An explicit selector is required") }
        let root = AXUIElementCreateApplication(app.processIdentifier)
        return descendants(root, limit: 500).filter { item in
            !secure(item) && (role == nil || string(item, kAXRoleAttribute).replacingOccurrences(of: "AX", with: "").caseInsensitiveCompare(role!.replacingOccurrences(of: "AX", with: "")) == .orderedSame)
                && (name == nil || label(item) == name) && (id == nil || string(item, kAXIdentifierAttribute) == id)
        }
    }

    private func unique(_ app: NSRunningApplication, _ selector: [String: Any]) throws -> AXUIElement {
        let matches = try matching(app, selector)
        guard matches.count == 1 else { throw failure("Semantic target is missing or ambiguous; observe again") }
        return matches[0]
    }

    private func descendants(_ root: AXUIElement, limit: Int) -> [AXUIElement] {
        AXUIElementSetMessagingTimeout(root, 0.3)
        let deadline = Date().addingTimeInterval(1)
        var queue: [(AXUIElement, Int)] = [(root, 0)]; var result: [AXUIElement] = []
        while !queue.isEmpty && result.count < limit && Date() < deadline {
            let (item, depth) = queue.removeFirst(); result.append(item)
            if depth < 8, let children = attribute(item, kAXChildrenAttribute) as? [AXUIElement] { queue.append(contentsOf: children.prefix(limit - result.count).map { ($0, depth + 1) }) }
        }
        return result
    }

    private func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
    }
    private func element(_ root: AXUIElement, _ name: String) -> AXUIElement? {
        guard let value = attribute(root, name), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return unsafeDowncast(value, to: AXUIElement.self)
    }
    private func string(_ element: AXUIElement, _ name: String) -> String { String((attribute(element, name) as? String ?? "").prefix(4000)) }
    private func label(_ element: AXUIElement) -> String { let title = string(element, kAXTitleAttribute); return title.isEmpty ? string(element, kAXDescriptionAttribute) : title }
    private func secure(_ element: AXUIElement) -> Bool { string(element, kAXSubroleAttribute).localizedCaseInsensitiveContains("secure") || string(element, kAXRoleAttribute).localizedCaseInsensitiveContains("password") }
    private func failure(_ text: String) -> Error { SageClientError.protocolError(text) }
}
