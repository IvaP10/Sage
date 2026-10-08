import AppKit
import ApplicationServices
import CryptoKit
import Foundation

/// A cancelled signature check may still be inside a synchronous system call.
/// Keep preparation on one actor so repeated transcripts cannot start an
/// unbounded set of inspections. Queued cancelled work is discarded on entry.
private actor ApplicationIdentityPreparation {
    func inspect(_ url: URL) throws -> Sage_Ipc_V2_ApplicationTarget {
        try Task.checkCancellation()
        let target = try SignedApplicationIdentity.inspect(url)
        try Task.checkCancellation()
        return target
    }
}

private struct FocusedReferenceObservation: Sendable {
    var accessibilityAvailable: Bool
    var hasWindow: Bool
    var windowTitle: String
    var resource: String
    var hasFocusedElement: Bool
    var selectedText: String
    var role: String
    var label: String
}

private enum InterfaceControlValue: Sendable, Equatable {
    case boolean(Bool)
    case number(Double)
}

private struct InterfaceControlObservation: Sendable {
    var identifier: String
    var role: String
    var label: String
    var enabled: Bool
    var ancestors: [String]
    var value: InterfaceControlValue?
    var step: Double?
    var minimum: Double?
    var maximum: Double?
}

private struct InterfaceDiscoveryObservation: Sendable {
    var accessibilityAvailable: Bool
    var windowTitle: String
    var controls: [InterfaceControlObservation]
    var truncated: Bool
}

private enum NativeProbeKind: String, Sendable {
    case restoreSliderValue = "restore_slider_value"
    case restoreToggleState = "restore_toggle_state"
}

private let maxInterfaceDiscoveryNodes = 512

private struct InterfaceProbeReadback: Sendable {
    var value: InterfaceControlValue
    var observedAt: Date
}

private struct InterfaceProbeReceipt: Sendable {
    var controlID: String
    var observations: [InterfaceProbeReadback]
    var restorationVerified: Bool
}

private struct InterfaceControlReadback: Sendable {
    var controlID: String
    var role: String
    var label: String
    var ancestors: [String]
    var enabled: Bool
    var value: InterfaceControlValue
    var observedAt: Date
}

/// Accessibility calls can block while another app is slow to answer. Keep
/// them off SwiftUI's main actor and serialize the OS queries.
private actor FocusedReferencePreparation {
    func inspect(processID: pid_t) -> FocusedReferenceObservation {
        guard AXIsProcessTrusted() else {
            return FocusedReferenceObservation(
                accessibilityAvailable: false, hasWindow: false, windowTitle: "", resource: "",
                hasFocusedElement: false, selectedText: "", role: "", label: ""
            )
        }
        let root = AXUIElementCreateApplication(processID)
        let window = element(root, kAXFocusedWindowAttribute)
        let focused = element(root, kAXFocusedUIElementAttribute)
        let safeFocus = focused.flatMap { secure($0) ? nil : $0 }
        return FocusedReferenceObservation(
            accessibilityAvailable: true,
            hasWindow: window != nil,
            windowTitle: window.map { string($0, kAXTitleAttribute) } ?? "",
            resource: window.map { string($0, kAXDocumentAttribute) } ?? "",
            hasFocusedElement: safeFocus != nil,
            selectedText: safeFocus.map { string($0, kAXSelectedTextAttribute) } ?? "",
            role: safeFocus.map { string($0, kAXRoleAttribute) } ?? "",
            label: safeFocus.map(label) ?? ""
        )
    }

    private func attribute(_ element: AXUIElement, _ name: String) -> Any? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else {
            return nil
        }
        return value
    }

    private func element(_ root: AXUIElement, _ name: String) -> AXUIElement? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(root, name as CFString, &value) == .success,
              let value,
              CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return unsafeDowncast(value, to: AXUIElement.self)
    }

    private func string(_ element: AXUIElement, _ name: String) -> String {
        String((attribute(element, name) as? String ?? "").prefix(4000))
    }

    private func label(_ element: AXUIElement) -> String {
        let title = string(element, kAXTitleAttribute)
        return title.isEmpty ? string(element, kAXDescriptionAttribute) : title
    }

    private func secure(_ element: AXUIElement) -> Bool {
        string(element, kAXSubroleAttribute).localizedCaseInsensitiveContains("secure")
            || string(element, kAXRoleAttribute).localizedCaseInsensitiveContains("password")
    }
}

/// Bounded, read-only accessibility discovery. It never invokes a control,
/// reads secure descendants, selection contents, or non-visible elements.
private actor InterfaceDiscoveryPreparation {
    func inspect(processID: pid_t) -> InterfaceDiscoveryObservation {
        guard AXIsProcessTrusted() else {
            return InterfaceDiscoveryObservation(
                accessibilityAvailable: false,
                windowTitle: "",
                controls: [],
                truncated: false
            )
        }
        let root = AXUIElementCreateApplication(processID)
        AXUIElementSetMessagingTimeout(root, 0.25)
        guard let window = element(root, kAXFocusedWindowAttribute) else {
            return InterfaceDiscoveryObservation(
                accessibilityAvailable: true,
                windowTitle: "",
                controls: [],
                truncated: false
            )
        }
        AXUIElementSetMessagingTimeout(window, 0.25)
        let deadline = Date().addingTimeInterval(1.0)
        var queue: [(AXUIElement, [String], Int)] = [(window, [], 0)]
        var controls: [InterfaceControlObservation] = []
        var discoveredCount = 1
        while !queue.isEmpty && controls.count < 48
                && discoveredCount <= maxInterfaceDiscoveryNodes && Date() < deadline {
            let (item, ancestors, depth) = queue.removeFirst()
            if secure(item) { continue }
            let role = string(item, kAXRoleAttribute)
            let name = label(item)
            let semanticRole = discoverableRole(role)
            let visible = (attribute(item, "AXVisible") as? NSNumber)?.boolValue
            if visible == true, !name.isEmpty, let semanticRole {
                let identifier = string(item, kAXIdentifierAttribute)
                let semanticIdentity = identifier.isEmpty
                    ? (ancestors + [role, name]).joined(separator: "|")
                    : identifier
                let enabled = (attribute(item, "AXEnabled") as? NSNumber)?.boolValue ?? false
                let value = attribute(item, "AXValue").flatMap { raw -> InterfaceControlValue? in
                    guard let number = raw as? NSNumber else { return nil }
                    if CFGetTypeID(number) == CFBooleanGetTypeID() {
                        return semanticRole == "switch" || semanticRole == "checkbox"
                            ? .boolean(number.boolValue) : nil
                    }
                    if semanticRole == "checkbox" || semanticRole == "switch" {
                        guard number.intValue == 0 || number.intValue == 1 else { return nil }
                        return .boolean(number.intValue == 1)
                    }
                    guard semanticRole == "slider" else { return nil }
                    let value = number.doubleValue
                    return value.isFinite ? .number(value) : nil
                }
                let stepValue = semanticRole == "slider"
                    ? (attribute(item, "AXValueIncrement") as? NSNumber)?.doubleValue : nil
                let step = stepValue.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
                let minimumValue = semanticRole == "slider"
                    ? (attribute(item, "AXMinValue") as? NSNumber)?.doubleValue : nil
                let minimum = minimumValue.flatMap { $0.isFinite ? $0 : nil }
                let maximumValue = semanticRole == "slider"
                    ? (attribute(item, "AXMaxValue") as? NSNumber)?.doubleValue : nil
                let maximum = maximumValue.flatMap { $0.isFinite ? $0 : nil }
                controls.append(InterfaceControlObservation(
                    identifier: controlIdentityDigest(semanticIdentity),
                    role: semanticRole,
                    label: String(name.prefix(256)),
                    enabled: enabled,
                    ancestors: ancestors.suffix(4).map { String($0.prefix(64)) },
                    value: value,
                    step: step,
                    minimum: minimum,
                    maximum: maximum
                ))
            }
            guard depth < 8,
                  let children = attribute(item, kAXChildrenAttribute) as? [AXUIElement] else { continue }
            let ancestorNames = (ancestors + [String(name.prefix(64))]).suffix(4)
            for child in children.prefix(max(0, maxInterfaceDiscoveryNodes - discoveredCount)) {
                queue.append((child, Array(ancestorNames), depth + 1))
                discoveredCount += 1
            }
        }
        return InterfaceDiscoveryObservation(
            accessibilityAvailable: true,
            windowTitle: String(string(window, kAXTitleAttribute).prefix(256)),
            controls: controls,
            truncated: !queue.isEmpty || controls.count >= 48
                || discoveredCount >= maxInterfaceDiscoveryNodes
        )
    }

    private func discoverableRole(_ role: String) -> String? {
        switch role {
        case "AXSlider": "slider"
        case "AXCheckBox": "checkbox"
        case "AXSwitch": "switch"
        case "AXButton": "button"
        case "AXRadioButton": "radio_button"
        case "AXPopUpButton": "popup_button"
        case "AXMenuButton": "menu_button"
        case "AXMenuItem": "menu_item"
        case "AXTab": "tab"
        case "AXLink": "link"
        case "AXDisclosureTriangle": "disclosure_triangle"
        default: nil
        }
    }

    /// Rebind one reviewed semantic anchor in the focused window, make a
    /// bounded reversible change, and restore the exact captured AXValue.
    /// This method intentionally contains no suspension point after dispatch:
    /// task cancellation cannot interrupt the required restoration attempt.
    func probe(processID: pid_t, controlID: String, kind: NativeProbeKind) throws -> InterfaceProbeReceipt {
        try Task.checkCancellation()
        guard AXIsProcessTrusted() else { throw probeFailure("Accessibility access is required for a learning probe") }
        guard controlID.count == 64,
              controlID.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw probeFailure("The semantic control identity is invalid")
        }
        let root = AXUIElementCreateApplication(processID)
        AXUIElementSetMessagingTimeout(root, 0.25)
        guard let window = element(root, kAXFocusedWindowAttribute) else {
            throw probeFailure("The approved application's focused window is unavailable")
        }
        AXUIElementSetMessagingTimeout(window, 0.25)
        let deadline = Date().addingTimeInterval(1.0)
        var queue: [(AXUIElement, [String], Int)] = [(window, [], 0)]
        var matches: [(AXUIElement, String, String)] = []
        var visited = 0
        while !queue.isEmpty && visited < 512 && Date() < deadline {
            let (item, ancestors, depth) = queue.removeFirst()
            visited += 1
            AXUIElementSetMessagingTimeout(item, 0.25)
            if !secure(item) {
                let role = string(item, kAXRoleAttribute)
                let name = label(item)
                let visible = (attribute(item, "AXVisible") as? NSNumber)?.boolValue == true
                if visible, !name.isEmpty,
                   ["AXSlider", "AXCheckBox", "AXSwitch"].contains(role) {
                    let automationID = string(item, kAXIdentifierAttribute)
                    let semanticIdentity = automationID.isEmpty
                        ? (ancestors + [role, name]).joined(separator: "|")
                        : automationID
                    if controlIdentityDigest(semanticIdentity) == controlID {
                        matches.append((item, role, name))
                    }
                }
            }
            guard depth < 8,
                  let children = attribute(item, kAXChildrenAttribute) as? [AXUIElement] else { continue }
            let ancestorNames = Array((ancestors + [label(item)]).suffix(6))
            for child in children.prefix(max(0, 512 - visited)) {
                queue.append((child, ancestorNames, depth + 1))
            }
        }
        guard Date() < deadline, matches.count == 1, let (control, role, controlLabel) = matches.first else {
            throw probeFailure("The approved control is missing, ambiguous, or the interface changed")
        }
        guard safeProbeLabel(controlLabel),
              (attribute(control, "AXEnabled") as? NSNumber)?.boolValue == true else {
            throw probeFailure("The approved control is disabled or outside the reversible probe policy")
        }
        let expectedRoleMatches = (kind == .restoreSliderValue && role == "AXSlider")
            || (kind == .restoreToggleState && (role == "AXCheckBox" || role == "AXSwitch"))
        guard expectedRoleMatches else { throw probeFailure("The approved probe kind no longer matches this control") }
        var settable = DarwinBoolean(false)
        guard AXUIElementIsAttributeSettable(control, kAXValueAttribute as CFString, &settable) == .success,
              settable.boolValue,
              let original = probeValue(attribute(control, kAXValueAttribute), role: role) else {
            throw probeFailure("The approved control does not expose a writable, typed value")
        }
        let beforeTime = Date()

        let initialValue: InterfaceControlValue
        let changedValue: InterfaceControlValue
        switch (kind, original) {
        case (.restoreToggleState, .boolean(let value)):
            initialValue = .boolean(value)
            changedValue = .boolean(!value)
        case (.restoreSliderValue, .number(let value)):
            guard let increment = (attribute(control, "AXValueIncrement") as? NSNumber)?.doubleValue,
                  let minimum = (attribute(control, "AXMinValue") as? NSNumber)?.doubleValue,
                  let maximum = (attribute(control, "AXMaxValue") as? NSNumber)?.doubleValue,
                  value.isFinite, increment.isFinite, increment > 0, increment <= 1,
                  minimum.isFinite, maximum.isFinite, minimum < maximum,
                  value >= minimum, value <= maximum else {
                throw probeFailure("The slider does not expose a bounded step of at most one unit")
            }
            let next = value + increment <= maximum ? value + increment : value - increment
            guard next.isFinite, next >= minimum, next <= maximum, next != value else {
                throw probeFailure("The slider has no safe adjacent value to test")
            }
            initialValue = .number(value)
            changedValue = .number(next)
        default:
            throw probeFailure("The approved control value type is not reversible")
        }

        try Task.checkCancellation()
        var restorationStillRequired = false
        defer {
            if restorationStillRequired {
                _ = AXUIElementSetAttributeValue(control, kAXValueAttribute as CFString, axValue(initialValue))
            }
        }
        restorationStillRequired = true
        let setStatus = AXUIElementSetAttributeValue(control, kAXValueAttribute as CFString, axValue(changedValue))
        let changedReadback = probeValue(attribute(control, kAXValueAttribute), role: role)
        let changedTime = monotonicTimestamp(after: beforeTime)

        // Restoration runs even when the attempted mutation returned an AX
        // error; that return can be uncertain after an accessibility timeout.
        let restoreStatus = AXUIElementSetAttributeValue(control, kAXValueAttribute as CFString, axValue(initialValue))
        let restoredReadback = probeValue(attribute(control, kAXValueAttribute), role: role)
        let restoredTime = monotonicTimestamp(after: changedTime)
        let restorationVerified = restoreStatus == .success && restoredReadback == initialValue
        if restorationVerified { restorationStillRequired = false }
        guard setStatus == .success,
              let changedReadback,
              let restoredReadback,
              changedReadback == changedValue,
              changedReadback != initialValue,
              restorationVerified else {
            throw probeFailure("The control change or exact restoration could not be independently verified")
        }
        return InterfaceProbeReceipt(
            controlID: controlID,
            observations: [
                InterfaceProbeReadback(value: initialValue, observedAt: beforeTime),
                InterfaceProbeReadback(value: changedReadback, observedAt: changedTime),
                InterfaceProbeReadback(value: restoredReadback, observedAt: restoredTime),
            ],
            restorationVerified: true
        )
    }

    private func probeValue(_ raw: CFTypeRef?, role: String) -> InterfaceControlValue? {
        guard let number = raw as? NSNumber else { return nil }
        if role == "AXCheckBox" || role == "AXSwitch" {
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return .boolean(number.boolValue) }
            guard number.intValue == 0 || number.intValue == 1 else { return nil }
            return .boolean(number.intValue == 1)
        }
        let value = number.doubleValue
        return value.isFinite ? .number(value) : nil
    }

    func readControl(processID: pid_t, controlID: String) throws -> InterfaceControlReadback {
        let (control, role, name, ancestors) = try exactControl(processID: processID, controlID: controlID)
        guard let value = probeValue(attribute(control, kAXValueAttribute), role: axRole(for: role)) else {
            throw probeFailure("The learned control no longer exposes a supported typed value")
        }
        return InterfaceControlReadback(
            controlID: controlID,
            role: role,
            label: name,
            ancestors: ancestors,
            enabled: (attribute(control, "AXEnabled") as? NSNumber)?.boolValue ?? false,
            value: value,
            observedAt: Date()
        )
    }

    /// Apply one exact typed value through the reviewed accessibility anchor.
    /// The caller performs an additional, separate read request for verification.
    func setControlValue(
        processID: pid_t,
        controlID: String,
        expectedRole: String,
        expectedLabel: String,
        expectedAncestors: [String],
        value: InterfaceControlValue
    ) throws -> Bool {
        let (control, role, name, ancestors) = try exactControl(processID: processID, controlID: controlID)
        guard role == expectedRole,
              name == expectedLabel,
              ancestors == expectedAncestors,
              safeProbeLabel(name),
              (attribute(control, "AXEnabled") as? NSNumber)?.boolValue == true else {
            throw probeFailure("The learned control no longer matches its approved semantic anchor")
        }
        switch (role, value) {
        case ("slider", .number(let requested)):
            guard requested.isFinite,
                  let current = (attribute(control, kAXValueAttribute) as? NSNumber)?.doubleValue,
                  current.isFinite,
                  let step = (attribute(control, "AXValueIncrement") as? NSNumber)?.doubleValue,
                  step.isFinite, step > 0, step <= 1,
                  let minimum = (attribute(control, "AXMinValue") as? NSNumber)?.doubleValue,
                  minimum.isFinite,
                  let maximum = (attribute(control, "AXMaxValue") as? NSNumber)?.doubleValue,
                  maximum.isFinite, minimum < maximum,
                  requested >= minimum, requested <= maximum else {
                throw probeFailure("The current slider range does not support the requested value")
            }
            let grid = (requested - minimum) / step
            guard grid.isFinite, abs(grid - grid.rounded()) <= 1e-7 * max(abs(grid), 1) else {
                throw probeFailure("The requested slider value is outside its current step grid")
            }
        case ("checkbox", .boolean), ("switch", .boolean):
            break
        default:
            throw probeFailure("The requested value type does not match the learned control role")
        }
        var settable = DarwinBoolean(false)
        guard AXUIElementIsAttributeSettable(control, kAXValueAttribute as CFString, &settable) == .success,
              settable.boolValue else {
            throw probeFailure("The learned control is no longer writable")
        }
        // This return code is diagnostic only. The Rust verifier makes a new
        // adapter request and independently reads the post-dispatch value.
        return AXUIElementSetAttributeValue(
            control,
            kAXValueAttribute as CFString,
            axValue(value)
        ) == .success
    }

    private func exactControl(
        processID: pid_t,
        controlID: String
    ) throws -> (AXUIElement, String, String, [String]) {
        guard AXIsProcessTrusted(), controlID.count == 64,
              controlID.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw probeFailure("Accessibility permission or learned control identity is invalid")
        }
        let root = AXUIElementCreateApplication(processID)
        AXUIElementSetMessagingTimeout(root, 0.25)
        guard let window = element(root, kAXFocusedWindowAttribute) else {
            throw probeFailure("The signed application's focused window is unavailable")
        }
        AXUIElementSetMessagingTimeout(window, 0.25)
        let deadline = Date().addingTimeInterval(1)
        var queue: [(AXUIElement, [String], Int)] = [(window, [], 0)]
        var matches: [(AXUIElement, String, String, [String])] = []
        var visited = 0
        while !queue.isEmpty && visited < maxInterfaceDiscoveryNodes && Date() < deadline {
            let (item, ancestors, depth) = queue.removeFirst()
            visited += 1
            AXUIElementSetMessagingTimeout(item, 0.25)
            if !secure(item), let role = discoverableRole(string(item, kAXRoleAttribute)) {
                let name = label(item)
                let visible = (attribute(item, "AXVisible") as? NSNumber)?.boolValue == true
                if visible, !name.isEmpty, ["slider", "checkbox", "switch"].contains(role) {
                    let identifier = string(item, kAXIdentifierAttribute)
                    let identity = identifier.isEmpty
                        ? (ancestors + [string(item, kAXRoleAttribute), name]).joined(separator: "|")
                        : identifier
                    if controlIdentityDigest(identity) == controlID {
                        matches.append((item, role, String(name.prefix(256)), Array(ancestors.suffix(4))))
                    }
                }
            }
            guard depth < 8,
                  let children = attribute(item, kAXChildrenAttribute) as? [AXUIElement] else { continue }
            let ancestorNames = Array((ancestors + [String(label(item).prefix(64))]).suffix(4))
            for child in children.prefix(max(0, maxInterfaceDiscoveryNodes - visited)) {
                queue.append((child, ancestorNames, depth + 1))
            }
        }
        guard Date() < deadline, matches.count == 1, let exact = matches.first else {
            throw probeFailure("The learned control is missing or ambiguous in the focused window")
        }
        return exact
    }

    private func axRole(for semanticRole: String) -> String {
        switch semanticRole {
        case "checkbox": "AXCheckBox"
        case "switch": "AXSwitch"
        default: "AXSlider"
        }
    }

    private func axValue(_ value: InterfaceControlValue) -> CFTypeRef {
        switch value {
        case .boolean(let value): NSNumber(value: value)
        case .number(let value): NSNumber(value: value)
        }
    }

    private func safeProbeLabel(_ label: String) -> Bool {
        let normalized = label.lowercased()
        let denied = [
            "send", "submit", "delete", "remove", "purchase", "buy", "pay", "publish",
            "security", "password", "privacy", "account", "install", "uninstall", "reset",
            "erase", "wipe", "logout", "sign out", "share", "connect", "disconnect",
            "microphone", "camera", "location", "firewall", "encryption",
        ]
        return !label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !denied.contains(where: normalized.contains)
    }

    private func monotonicTimestamp(after previous: Date) -> Date {
        let current = Date()
        return current > previous.addingTimeInterval(0.002)
            ? current
            : previous.addingTimeInterval(0.002)
    }

    private func probeFailure(_ message: String) -> Error {
        SageClientError.protocolError(message)
    }

    private func controlIdentityDigest(_ identity: String) -> String {
        SHA256.hash(data: Data(identity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }

    private func element(_ root: AXUIElement, _ name: String) -> AXUIElement? {
        guard let value = attribute(root, name), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return unsafeDowncast(value, to: AXUIElement.self)
    }

    private func string(_ element: AXUIElement, _ name: String) -> String {
        String((attribute(element, name) as? String ?? "").prefix(512))
    }

    private func label(_ element: AXUIElement) -> String {
        let title = string(element, kAXTitleAttribute)
        return title.isEmpty ? string(element, kAXDescriptionAttribute) : title
    }

    private func secure(_ element: AXUIElement) -> Bool {
        string(element, kAXSubroleAttribute).localizedCaseInsensitiveContains("secure")
            || string(element, kAXRoleAttribute).localizedCaseInsensitiveContains("password")
    }
}

/// Native OS access lives in the client; only an authenticated core request can
/// reach this adapter. It never plans, approves, or grants filesystem access.
@MainActor
final class PlatformAdapter {
    private var previousApplication: NSRunningApplication?
    private var activationObserver: NSObjectProtocol?
    private var usedGrants: [String: Date] = [:]
    private let applicationPreparation = ApplicationIdentityPreparation()
    private let referencePreparation = FocusedReferencePreparation()
    private let discoveryPreparation = InterfaceDiscoveryPreparation()

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
            try Task.checkCancellation()
            guard request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else { throw failure("Adapter request expired") }
            let payload = try JSONSerialization.jsonObject(with: Data(request.json.utf8)) as? [String: Any] ?? [:]
            let data: [String: Any]
            switch request.operation {
            case "reference": data = await reference()
            case "observe": data = try observe(payload)
            case "application_identity":
                let identifier = payload["application"] as? String ?? ""
                let url = try resolveApplication(identifier)
                result.applicationTarget = try await applicationPreparation.inspect(url)
                data = [:]
            case "observe_application":
                guard request.hasApplicationTarget else { throw failure("Missing typed application identity") }
                let target = request.applicationTarget
                let app = try exactRunning(target)
                let pid = app.processIdentifier
                try await Task.detached(priority: .userInitiated) { try SignedApplicationIdentity.validateProcess(pid, target: target) }.value
                result.applicationTarget = target
                result.observedProcessID = UInt32(pid)
                data = [:]
            case "observe_application_control":
                guard request.hasApplicationTarget else { throw failure("Missing typed application identity") }
                let target = request.applicationTarget
                let controlID = payload["control_id"] as? String ?? ""
                guard let active = currentExternalApplication(),
                      active.bundleIdentifier == target.identifier,
                      active.processIdentifier > 0 else {
                    throw failure("The learned application's foreground target changed before readback")
                }
                let app = try exactRunning(target)
                let processID = app.processIdentifier
                guard active.processIdentifier == processID else {
                    throw failure("The learned application's process changed before readback")
                }
                try await Task.detached(priority: .userInitiated) {
                    try SignedApplicationIdentity.validateProcess(processID, target: target)
                }.value
                let readback = try await discoveryPreparation.readControl(
                    processID: processID,
                    controlID: controlID
                )
                guard currentExternalApplication()?.processIdentifier == processID,
                      NSRunningApplication(processIdentifier: processID)?.isTerminated == false else {
                    throw failure("The learned foreground target changed before readback completed")
                }
                result.applicationTarget = target
                result.observedProcessID = UInt32(processID)
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                data = [
                    "control_id": readback.controlID,
                    "role": readback.role,
                    "label": readback.label,
                    "ancestors": readback.ancestors,
                    "enabled": readback.enabled,
                    "value": interfaceValueJSON(readback.value),
                    "observed_at": formatter.string(from: readback.observedAt),
                ]
            case "discover_interface":
                let app = currentExternalApplication()
                guard let app,
                      app.processIdentifier != ProcessInfo.processInfo.processIdentifier,
                      !app.isTerminated,
                      let bundleURL = app.bundleURL else {
                    throw failure("There is no current external application to inspect")
                }
                let processID = app.processIdentifier
                let target = try await applicationPreparation.inspect(bundleURL)
                try await Task.detached(priority: .userInitiated) {
                    try SignedApplicationIdentity.validateProcess(processID, target: target)
                }.value
                let observation = await discoveryPreparation.inspect(processID: processID)
                try await Task.detached(priority: .userInitiated) {
                    try SignedApplicationIdentity.validateProcess(processID, target: target)
                }.value
                let visibleApp = NSWorkspace.shared.frontmostApplication
                let stillCurrent = visibleApp?.processIdentifier == processID
                    || (visibleApp?.processIdentifier == ProcessInfo.processInfo.processIdentifier
                        && previousApplication?.processIdentifier == processID)
                guard stillCurrent,
                      NSRunningApplication(processIdentifier: processID)?.isTerminated == false else {
                    throw failure("The foreground application changed during passive discovery")
                }
                result.applicationTarget = target
                result.observedProcessID = UInt32(processID)
                data = [
                    "application_name": app.localizedName ?? target.identifier,
                    "bundle_identifier": target.identifier,
                    "accessibility_available": observation.accessibilityAvailable,
                    "active_window": observation.windowTitle,
                    "truncated": observation.truncated,
                    "controls": observation.controls.map { control in
                        var value: [String: Any] = [
                            "id": control.identifier,
                            "role": control.role,
                            "label": control.label,
                            "enabled": control.enabled,
                            "ancestors": control.ancestors,
                        ]
                        if let controlValue = control.value {
                            switch controlValue {
                            case .boolean(let boolean): value["value"] = boolean
                            case .number(let number): value["value"] = number
                            }
                        }
                        if let step = control.step { value["step"] = step }
                        if let minimum = control.minimum { value["minimum"] = minimum }
                        if let maximum = control.maximum { value["maximum"] = maximum }
                        return value
                    },
                ]
            case "probe_control":
                guard request.hasApplicationTarget else { throw failure("Missing signed application identity for learning probe") }
                let target = request.applicationTarget
                let lease = payload["probe_lease"] as? [String: Any] ?? [:]
                let leaseID = lease["id"] as? String ?? ""
                let sessionID = lease["session_id"] as? String ?? ""
                let systemID = lease["system_id"] as? String ?? ""
                let workerSession = lease["worker_session"] as? String ?? ""
                let fingerprint = lease["system_fingerprint"] as? String ?? ""
                let controlID = lease["control_id"] as? String ?? ""
                let kindText = lease["kind"] as? String ?? ""
                let expectedProcessID = (payload["expected_process_id"] as? NSNumber)?.int32Value
                let leasedProcessID = (lease["expected_process_id"] as? NSNumber)?.int32Value
                let issuedAt = parseDate(lease["issued_at"] as? String ?? "")
                let expiresAt = parseDate(lease["expires_at"] as? String ?? "")
                guard UUID(uuidString: leaseID) != nil,
                      UUID(uuidString: sessionID) != nil,
                      UUID(uuidString: systemID) != nil,
                      !workerSession.isEmpty,
                      fingerprint == target.codeDigest,
                      controlID.count == 64,
                      controlID.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
                      let kind = NativeProbeKind(rawValue: kindText),
                      let expectedProcessID, expectedProcessID > 0,
                      let leasedProcessID,
                      expectedProcessID == leasedProcessID,
                      let issuedAt, let expiresAt,
                      issuedAt <= Date(), expiresAt > Date(),
                      request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else {
                    throw failure("Learning probe lease is malformed, expired, or bound to a different app")
                }
                guard let activeApp = currentExternalApplication(),
                      activeApp.processIdentifier != ProcessInfo.processInfo.processIdentifier,
                      activeApp.bundleIdentifier == target.identifier,
                      activeApp.processIdentifier == expectedProcessID else {
                    throw failure("The approved application is no longer the current foreground target")
                }
                let app = try exactRunning(target)
                let processID = app.processIdentifier
                guard activeApp.processIdentifier == processID,
                      processID == expectedProcessID else {
                    throw failure("The approved application process changed before the probe")
                }
                try await Task.detached(priority: .userInitiated) {
                    try SignedApplicationIdentity.validateProcess(processID, target: target)
                }.value
                try Task.checkCancellation()
                let receipt = try await discoveryPreparation.probe(
                    processID: processID,
                    controlID: controlID,
                    kind: kind
                )
                try await Task.detached(priority: .userInitiated) {
                    try SignedApplicationIdentity.validateProcess(processID, target: target)
                }.value
                guard currentExternalApplication()?.processIdentifier == processID,
                      NSRunningApplication(processIdentifier: processID)?.isTerminated == false else {
                    throw failure("The foreground application changed while the probe was settling")
                }
                result.applicationTarget = target
                result.observedProcessID = UInt32(processID)
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                data = [
                    "control_id": receipt.controlID,
                    "restoration_verified": receipt.restorationVerified,
                    "observations": receipt.observations.map { readback in
                        let value: Any
                        switch readback.value {
                        case .boolean(let boolean): value = boolean
                        case .number(let number): value = number
                        }
                        return [
                            "value": value,
                            "observed_at": formatter.string(from: readback.observedAt),
                        ]
                    },
                ]
            case "execute":
                let (app, output) = try await execute(request, payload)
                result.applicationTarget = request.applicationTarget
                result.observedProcessID = UInt32(app.processIdentifier)
                data = output
            default: throw failure("Unknown adapter operation")
            }
            result.json = String(decoding: try JSONSerialization.data(withJSONObject: data), as: UTF8.self)
            result.success = true
        } catch { result.error = error.localizedDescription }
        return result
    }

    private func currentExternalApplication() -> NSRunningApplication? {
        let frontmost = NSWorkspace.shared.frontmostApplication
        return frontmost?.processIdentifier == ProcessInfo.processInfo.processIdentifier
            ? previousApplication
            : frontmost
    }

    private func parseDate(_ value: String) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        for suffix in [".SSSSSSSSSXXXXX", ".SSSXXXXX", "XXXXX"] {
            formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss\(suffix)"
            if let date = formatter.date(from: value) { return date }
        }
        return nil
    }

    /// Minimal foreground fields used only when the user's request explicitly
    /// refers to a current selection, document, window, or page.
    private func reference() async -> [String: Any] {
        let front = NSWorkspace.shared.frontmostApplication
        let app = front?.processIdentifier == ProcessInfo.processInfo.processIdentifier ? previousApplication : front
        guard let app else { return ["available": false, "reason": "No foreground application is available."] }
        var result: [String: Any] = [
            "active_application": app.bundleIdentifier ?? "",
            "application_name": app.localizedName ?? "Current app",
            "available": false,
        ]
        let observation = await referencePreparation.inspect(processID: app.processIdentifier)
        guard observation.accessibilityAvailable else {
            result["reason"] = "Accessibility access is off."
            return result
        }
        if observation.hasWindow {
            result["active_window"] = observation.windowTitle
            result["current_resource"] = observation.resource
            result["available"] = true
        }
        if observation.hasFocusedElement {
            result["selected_text"] = observation.selectedText
            result["selection"] = [
                "role": observation.role,
                "label": observation.label,
            ]
            result["available"] = true
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

    private func resolveApplication(_ identifier: String) throws -> URL {
        guard !identifier.isEmpty, identifier.count <= 256,
              !identifier.contains("/"), !identifier.contains("\\"),
              !identifier.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
              identifier != Bundle.main.bundleIdentifier else { throw failure("Specify an installed application name or bundle identifier") }
        let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: identifier)
            ?? ["/Applications", "/System/Applications"].map { URL(fileURLWithPath: $0).appendingPathComponent(identifier + ".app") }
                .first { FileManager.default.fileExists(atPath: $0.path) }
        guard let url else { throw failure("Application could not be resolved to an installed bundle") }
        return url
    }

    private func exactRunning(_ target: Sage_Ipc_V2_ApplicationTarget) throws -> NSRunningApplication {
        let candidates = NSRunningApplication.runningApplications(withBundleIdentifier: target.identifier)
        guard candidates.count == 1, let app = candidates.first,
              app.bundleURL?.resolvingSymlinksInPath().standardizedFileURL.path == target.bundlePath,
              app.processIdentifier != ProcessInfo.processInfo.processIdentifier, !app.isTerminated else {
            throw failure("Running application is missing, ambiguous or differs from the approved bundle")
        }
        return app
    }

    private func execute(
        _ request: Sage_Ipc_V2_AdapterRequest,
        _ payload: [String: Any]
    ) async throws -> (NSRunningApplication, [String: Any]) {
        let action = payload["action"] as? [String: Any] ?? [:]
        let grant = request.grant
        let target = request.applicationTarget
        let actionKind = action["type"] as? String ?? ""
        guard request.hasGrant, request.hasApplicationTarget, grant.policyVersion == 2, grant.domain == "native",
              UUID(uuidString: grant.grantID) != nil, UUID(uuidString: grant.runID) != nil,
              UUID(uuidString: grant.actionID) != nil, UUID(uuidString: grant.workerSession) != nil,
              grant.operations.count == 2, Set(grant.operations) == ["observe", "control"],
              grant.actionDigest.utf8.count == 64,
              grant.actionDigest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              ["open_application", "set_application_control"].contains(actionKind),
              action["application"] as? String == target.identifier,
              target.identifier != Bundle.main.bundleIdentifier,
              case .application(let identifier) = grant.resource, identifier == target.identifier else {
            throw failure("Typed application grant does not match the prepared action")
        }
        let expires = Date(timeIntervalSince1970: Double(grant.expiresAtUnixMs) / 1000)
        usedGrants = usedGrants.filter { $0.value > Date() }
        guard expires > Date(), usedGrants[grant.grantID] == nil else { throw failure("Application grant expired or was already used") }
        // Consume before validation or dispatch. A failed attempt needs a fresh grant.
        usedGrants[grant.grantID] = expires
        try await Task.detached(priority: .userInitiated) { try SignedApplicationIdentity.validate(target) }.value
        guard !Task.isCancelled, expires > Date(), request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else {
            throw failure("Application grant expired during validation")
        }
        if actionKind == "set_application_control" {
            return try await executeApplicationControl(request, payload, action, target, expires)
        }
        if !NSRunningApplication.runningApplications(withBundleIdentifier: target.identifier).isEmpty {
            let app = try exactRunning(target)
            let pid = app.processIdentifier
            try await Task.detached(priority: .userInitiated) { try SignedApplicationIdentity.validateProcess(pid, target: target) }.value
            guard !Task.isCancelled, expires > Date(), request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else {
                throw failure("Application grant expired during process validation")
            }
            guard app.activate() else { throw failure("Application did not activate") }
            return (app, [:])
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.createsNewApplicationInstance = false
        // No arguments, environment, URLs or documents can be supplied by the model.
        let app = try await NSWorkspace.shared.openApplication(at: URL(fileURLWithPath: target.bundlePath), configuration: configuration)
        guard app.bundleIdentifier == target.identifier,
              app.bundleURL?.resolvingSymlinksInPath().standardizedFileURL.path == target.bundlePath else {
            throw failure("Launch result differs from the approved bundle")
        }
        return (app, [:])
    }

    private func executeApplicationControl(
        _ request: Sage_Ipc_V2_AdapterRequest,
        _ payload: [String: Any],
        _ action: [String: Any],
        _ target: Sage_Ipc_V2_ApplicationTarget,
        _ grantExpiry: Date
    ) async throws -> (NSRunningApplication, [String: Any]) {
        let systemID = action["system_id"] as? String ?? ""
        let systemFingerprint = action["system_fingerprint"] as? String ?? ""
        let capabilityID = action["capability_id"] as? String ?? ""
        let controlID = action["control_id"] as? String ?? ""
        let interfaceFingerprint = payload["application_interface_fingerprint"] as? String ?? ""
        let anchor = payload["application_control_anchor"] as? [String: Any] ?? [:]
        let anchorID = anchor["id"] as? String ?? ""
        let role = anchor["role"] as? String ?? ""
        let label = anchor["label"] as? String ?? ""
        let ancestors = anchor["ancestors"] as? [String] ?? []
        let enabled = anchor["enabled"] as? Bool ?? false
        guard UUID(uuidString: systemID) != nil,
              systemFingerprint == target.codeDigest,
              validControlDigest(interfaceFingerprint),
              validControlDigest(controlID),
              anchorID == controlID,
              !capabilityID.isEmpty, capabilityID.utf8.count <= 128,
              capabilityID.utf8.allSatisfy({
                  (97...122).contains($0) || (48...57).contains($0) || [46, 95, 45, 58].contains($0)
              }),
              enabled,
              ["slider", "checkbox", "switch"].contains(role),
              !label.isEmpty, label.utf8.count <= 256,
              ancestors.count <= 4,
              ancestors.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 64 }),
              let requested = interfaceControlValue(action["value"]) else {
            throw failure("Learned-control action or reviewed semantic anchor is malformed")
        }
        guard (role == "slider" && isNumber(requested))
                || ((role == "checkbox" || role == "switch") && isBoolean(requested)) else {
            throw failure("Learned-control value type differs from the reviewed control")
        }
        guard let active = currentExternalApplication(),
              active.bundleIdentifier == target.identifier,
              active.processIdentifier != ProcessInfo.processInfo.processIdentifier,
              active.processIdentifier > 0 else {
            throw failure("The learned application is no longer the current foreground target")
        }
        let app = try exactRunning(target)
        let processID = app.processIdentifier
        guard active.processIdentifier == processID,
              !app.isTerminated,
              !Task.isCancelled,
              grantExpiry > Date(),
              request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else {
            throw failure("The approved foreground application or one-use grant changed before dispatch")
        }
        try await Task.detached(priority: .userInitiated) {
            try SignedApplicationIdentity.validateProcess(processID, target: target)
        }.value
        guard currentExternalApplication()?.processIdentifier == processID,
              grantExpiry > Date(),
              request.expiresAtUnixMs > Int64(Date().timeIntervalSince1970 * 1000) else {
            throw failure("The approved foreground application changed during target validation")
        }
        let setSucceeded = try await discoveryPreparation.setControlValue(
            processID: processID,
            controlID: controlID,
            expectedRole: role,
            expectedLabel: label,
            expectedAncestors: ancestors,
            value: requested
        )
        return (app, [
            "control_id": controlID,
            "set_call_succeeded": setSucceeded,
        ])
    }

    private func validControlDigest(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }

    private func interfaceControlValue(_ value: Any?) -> InterfaceControlValue? {
        guard let number = value as? NSNumber else { return nil }
        if CFGetTypeID(number) == CFBooleanGetTypeID() {
            return .boolean(number.boolValue)
        }
        let double = number.doubleValue
        return double.isFinite ? .number(double) : nil
    }

    private func isNumber(_ value: InterfaceControlValue) -> Bool {
        if case .number = value { return true }
        return false
    }

    private func isBoolean(_ value: InterfaceControlValue) -> Bool {
        if case .boolean = value { return true }
        return false
    }

    private func interfaceValueJSON(_ value: InterfaceControlValue) -> Any {
        switch value {
        case .boolean(let boolean): boolean
        case .number(let number): number
        }
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
