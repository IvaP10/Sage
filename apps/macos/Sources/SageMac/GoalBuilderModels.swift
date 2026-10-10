import Foundation
import CoreFoundation

struct GoalSystemOption: Identifiable, Hashable {
    let id: String
    let label: String
    let kind: String
    let fingerprint: String

    init?(_ value: Any) {
        guard let object = value as? [String: Any],
              let id = object["id"] as? String,
              UUID(uuidString: id) != nil,
              let label = object["label"] as? String,
              let kind = object["kind"] as? String,
              let fingerprint = object["fingerprint"] as? String,
              isCanonicalSHA256(fingerprint) else { return nil }
        self.id = id
        self.label = label
        self.kind = kind
        self.fingerprint = fingerprint
    }
}

struct GoalPortOption: Hashable, Identifiable {
    let name: String
    let valueType: String
    let maximumBytes: Int
    let privacy: String

    var id: String { name }

    init?(_ value: Any) {
        guard let object = value as? [String: Any],
              let name = object["name"] as? String,
              let valueType = object["value_type"] as? String,
              let byteNumber = object["max_bytes"] as? NSNumber,
              CFGetTypeID(byteNumber) != CFBooleanGetTypeID(),
              byteNumber.doubleValue.isFinite,
              byteNumber.doubleValue > 0,
              byteNumber.doubleValue.rounded(.towardZero) == byteNumber.doubleValue,
              let byteLimit = Int(exactly: byteNumber.int64Value),
              byteLimit > 0,
              NSNumber(value: byteNumber.int64Value).compare(byteNumber) == .orderedSame,
              let privacy = object["privacy"] as? String else { return nil }
        self.name = name
        self.valueType = valueType
        self.maximumBytes = byteLimit
        self.privacy = privacy
    }

    var wireObject: [String: Any] {
        [
            "name": name,
            "value_type": valueType,
            "max_bytes": maximumBytes,
            "privacy": privacy,
        ]
    }
}

struct GoalCapabilityOption: Identifiable, Hashable {
    let id: String
    let systemID: String
    let systemFingerprint: String
    let label: String
    let input: GoalPortOption
    let output: GoalPortOption
    let effects: [String]
    let verification: String
    let restoration: String

    init?(_ value: Any, for system: GoalSystemOption) {
        guard let assessment = value as? [String: Any],
              assessment["evidence_state"] as? String == "reversibly_experimented",
              let descriptor = assessment["descriptor"] as? [String: Any],
              let id = descriptor["id"] as? String,
              let systemID = descriptor["system_id"] as? String,
              systemID == system.id,
              let fingerprint = descriptor["system_fingerprint"] as? String,
              fingerprint == system.fingerprint,
              descriptor["executor_id"] as? String == "set_application_control",
              let probeKind = descriptor["interface_probe_kind"] as? String,
              ["restore_slider_value", "restore_toggle_state"].contains(probeKind),
              let effects = descriptor["effects"] as? [String],
              Set(effects) == ["control_application"],
              let inputValues = descriptor["input_ports"] as? [Any],
              inputValues.count == 1,
              let input = GoalPortOption(inputValues[0]),
              input.name == "value",
              let outputValues = descriptor["output_ports"] as? [Any],
              outputValues.count == 1,
              let output = GoalPortOption(outputValues[0]),
              output.name == "observed_value",
              let label = descriptor["label"] as? String,
              let verification = descriptor["verification"] as? String,
              let restoration = descriptor["restoration"] as? String,
              !restoration.isEmpty else { return nil }

        let expectedType = probeKind == "restore_slider_value" ? "number" : "boolean"
        guard input.valueType == expectedType,
              output.valueType == expectedType,
              input.privacy == "private", output.privacy == "private",
              input.maximumBytes == 8, output.maximumBytes == 8 else { return nil }

        self.id = id
        self.systemID = systemID
        self.systemFingerprint = fingerprint
        self.label = label
        self.input = input
        self.output = output
        self.effects = effects.sorted()
        self.verification = verification
        self.restoration = restoration
    }
}

struct GoalPlanStep: Identifiable, Hashable {
    let id: String
    let capabilityID: String
    let capabilityLabel: String
    let inputSummary: String
    let effects: [String]
    let verification: String
    let restoration: String
}

struct GoalPlanPreview: Hashable {
    let steps: [GoalPlanStep]
    let waves: [[String]]
    let estimatedElapsedMicros: UInt64
}

enum GoalLiteralValue {
    case number(Double)
    case boolean(Bool)
}

private func isCanonicalSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { byte in
        (byte >= 48 && byte <= 57) || (byte >= 97 && byte <= 102)
    }
}
