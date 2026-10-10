import XCTest
@testable import SageMac

final class GoalBuilderTests: XCTestCase {
    func testOnlyCurrentReversiblePrivateScalarApplicationControlsAreSelectable() {
        let system = GoalSystemOption([
            "id": "11111111-1111-4111-8111-111111111111",
            "label": "Example Editor",
            "kind": "application",
            "fingerprint": String(repeating: "a", count: 64),
        ])!
        let port: [String: Any] = [
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
            "system_id": system.id,
            "system_fingerprint": system.fingerprint,
            "executor_id": "set_application_control",
            "interface_probe_kind": "restore_slider_value",
            "effects": ["control_application"],
            "input_ports": [port],
            "output_ports": [output],
            "label": "Set Brightness",
            "verification": "Fresh OS value readback",
            "restoration": "Restore the exact initial value and verify",
        ]
        let assessment: [String: Any] = [
            "evidence_state": "reversibly_experimented",
            "descriptor": descriptor,
        ]

        XCTAssertNotNil(GoalCapabilityOption(assessment, for: system))

        var untested = assessment
        untested["evidence_state"] = "passively_observed"
        XCTAssertNil(GoalCapabilityOption(untested, for: system))

        var unsafe = assessment
        var unsafeDescriptor = descriptor
        unsafeDescriptor["effects"] = ["control_application", "delete"]
        unsafe["descriptor"] = unsafeDescriptor
        XCTAssertNil(GoalCapabilityOption(unsafe, for: system))

        var incomplete = assessment
        var incompleteDescriptor = descriptor
        incompleteDescriptor.removeValue(forKey: "restoration")
        incomplete["descriptor"] = incompleteDescriptor
        XCTAssertNil(GoalCapabilityOption(incomplete, for: system))

        var stale = assessment
        var staleDescriptor = descriptor
        staleDescriptor["system_fingerprint"] = String(repeating: "b", count: 64)
        stale["descriptor"] = staleDescriptor
        XCTAssertNil(GoalCapabilityOption(stale, for: system))

        var fractionalPort = assessment
        var fractionalDescriptor = descriptor
        var fractionalInput = port
        fractionalInput["max_bytes"] = NSNumber(value: 8.5)
        fractionalDescriptor["input_ports"] = [fractionalInput]
        fractionalPort["descriptor"] = fractionalDescriptor
        XCTAssertNil(GoalCapabilityOption(fractionalPort, for: system))

        var booleanPort = assessment
        var booleanDescriptor = descriptor
        var booleanInput = port
        booleanInput["max_bytes"] = true
        booleanDescriptor["input_ports"] = [booleanInput]
        booleanPort["descriptor"] = booleanDescriptor
        XCTAssertNil(GoalCapabilityOption(booleanPort, for: system))

        XCTAssertNil(GoalSystemOption([
            "id": system.id,
            "label": system.label,
            "kind": system.kind,
            "fingerprint": String(repeating: "A", count: 64),
        ]))
    }
}
