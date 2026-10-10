import Foundation

@main
enum GoalBuilderModelSmoke {
    static func main() {
        let systemID = "11111111-1111-4111-8111-111111111111"
        let fingerprint = String(repeating: "a", count: 64)
        guard let system = GoalSystemOption([
            "id": systemID,
            "label": "Example Editor",
            "kind": "application",
            "fingerprint": fingerprint,
        ]) else {
            fatalError("valid system evidence was rejected")
        }

        let input: [String: Any] = [
            "name": "value",
            "value_type": "number",
            "max_bytes": 8,
            "privacy": "private",
        ]
        let output: [String: Any] = [
            "name": "observed_value",
            "value_type": "number",
            "max_bytes": 8,
            "privacy": "private",
        ]
        let descriptor: [String: Any] = [
            "id": "ui.control.slider.0123456789abcdef01234567",
            "system_id": systemID,
            "system_fingerprint": fingerprint,
            "executor_id": "set_application_control",
            "interface_probe_kind": "restore_slider_value",
            "effects": ["control_application"],
            "input_ports": [input],
            "output_ports": [output],
            "label": "Set Brightness",
            "verification": "Fresh OS value readback",
            "restoration": "Restore the exact initial value and verify",
        ]
        let assessment: [String: Any] = [
            "evidence_state": "reversibly_experimented",
            "descriptor": descriptor,
        ]
        precondition(GoalCapabilityOption(assessment, for: system) != nil,
                     "valid reversible private scalar capability was rejected")

        var fractionalAssessment = assessment
        var fractionalDescriptor = descriptor
        var fractionalInput = input
        fractionalInput["max_bytes"] = NSNumber(value: 8.5)
        fractionalDescriptor["input_ports"] = [fractionalInput]
        fractionalAssessment["descriptor"] = fractionalDescriptor
        precondition(GoalCapabilityOption(fractionalAssessment, for: system) == nil,
                     "fractional byte limit was accepted")

        var booleanAssessment = assessment
        var booleanDescriptor = descriptor
        var booleanInput = input
        booleanInput["max_bytes"] = true
        booleanDescriptor["input_ports"] = [booleanInput]
        booleanAssessment["descriptor"] = booleanDescriptor
        precondition(GoalCapabilityOption(booleanAssessment, for: system) == nil,
                     "boolean byte limit was accepted as an integer")

        precondition(GoalSystemOption([
            "id": systemID,
            "label": "Example Editor",
            "kind": "application",
            "fingerprint": String(repeating: "A", count: 64),
        ]) == nil, "noncanonical system fingerprint was accepted")
        print("Goal builder typed metadata smoke checks passed")
    }
}
