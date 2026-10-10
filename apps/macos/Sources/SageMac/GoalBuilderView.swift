import SwiftUI

struct GoalBuilderView: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var numberValue = ""
    @State private var toggleValue = ""

    private var selectedCapability: GoalCapabilityOption? {
        model.goalCapabilities.first { $0.id == model.selectedGoalCapabilityID }
    }

    private var numberIsValid: Bool {
        guard let number = Double(numberValue.trimmingCharacters(in: .whitespacesAndNewlines)) else { return false }
        return number.isFinite
    }

    private var canPreview: Bool {
        guard !model.worldModelBusy,
              !model.goalSubmissionNeedsRetry,
              let capability = selectedCapability else { return false }
        if capability.input.valueType == "number" { return numberIsValid }
        return toggleValue == "true" || toggleValue == "false"
    }

    private var resolvedTaskDescription: String {
        guard let capability = selectedCapability else { return "" }
        let input: String
        if capability.input.valueType == "number" {
            input = numberValue.trimmingCharacters(in: .whitespacesAndNewlines)
        } else if toggleValue == "true" {
            input = "on"
        } else if toggleValue == "false" {
            input = "off"
        } else {
            return ""
        }
        return "Set \(capability.label) to \(input)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Build a verified goal")
                    .font(.system(size: 20, weight: .semibold))
                Text("Choose an application control Sage has already tested and restored. Preview the typed procedure before starting it.")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if model.goalSubmissionNeedsRetry {
                retryPanel
            } else {
                systemSelection
                capabilitySelection
                if let capability = selectedCapability {
                    goalInputs(capability)
                }
                if let preview = model.goalPlanPreview {
                    previewPanel(preview)
                }
            }

            if !model.goalStatus.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    if model.worldModelBusy { ProgressView().controlSize(.small) }
                    Text(model.goalStatus)
                        .font(.system(size: 12))
                        .foregroundStyle(model.goalSubmissionNeedsRetry ? SageTheme.warning : Color.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Divider()
            HStack {
                Button("Later") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .disabled(model.worldModelBusy)
                Spacer()
                if model.goalSubmissionNeedsRetry {
                    Button("Retry exact submission") { model.retryGoalSubmission() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(model.worldModelBusy)
                } else if model.goalPlanPreview == nil {
                    Button("Preview procedure") { previewSelectedGoal() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(!canPreview)
                } else {
                    Button("Start and request approval") {
                        model.runPreparedGoal(taskDescription: resolvedTaskDescription)
                    }
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.worldModelBusy || resolvedTaskDescription.isEmpty)
                }
            }
        }
        .padding(24)
        .frame(minWidth: 520, idealWidth: 660, maxWidth: 760, minHeight: 420, idealHeight: 610, maxHeight: 760)
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled(model.worldModelBusy)
        .onAppear {
            numberValue = ""
            toggleValue = ""
        }
        .onChange(of: model.selectedGoalSystemID) {
            numberValue = ""
            toggleValue = ""
            model.goalSystemSelectionChanged()
        }
        .onChange(of: model.selectedGoalCapabilityID) {
            numberValue = ""
            toggleValue = ""
            model.invalidateGoalPreview()
        }
        .onChange(of: numberValue) { model.invalidateGoalPreview() }
        .onChange(of: toggleValue) { model.invalidateGoalPreview() }
    }

    private var systemSelection: some View {
        HStack(spacing: 10) {
            Picker("Application", selection: $model.selectedGoalSystemID) {
                ForEach(model.goalSystems) { system in
                    Text(system.label).tag(system.id)
                }
            }
            .labelsHidden()
            .accessibilityLabel("Application")
            .disabled(model.worldModelBusy || model.goalSystems.isEmpty)

            Button("Load current controls") { model.loadGoalCapabilities() }
                .buttonStyle(SageBorderedButtonStyle())
                .disabled(model.worldModelBusy || model.selectedGoalSystemID.isEmpty || model.goalSubmissionNeedsRetry)

            Button {
                model.refreshGoalSystems()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.plain)
            .help("Refresh discovered application systems")
            .accessibilityLabel("Refresh discovered applications")
            .disabled(model.worldModelBusy || model.goalSubmissionNeedsRetry)
        }
    }

    @ViewBuilder
    private var capabilitySelection: some View {
        if !model.goalCapabilities.isEmpty {
            Picker("Verified control", selection: $model.selectedGoalCapabilityID) {
                ForEach(model.goalCapabilities) { capability in
                    Text(capability.label).tag(capability.id)
                }
            }
            .disabled(model.worldModelBusy || model.goalSubmissionNeedsRetry)
        } else {
            Text("Only current controls with a verified restoration and a registered Sage executor can appear here.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func goalInputs(_ capability: GoalCapabilityOption) -> some View {
        VStack(alignment: .leading, spacing: 11) {
            if capability.input.valueType == "number" {
                TextField("Target number", text: $numberValue)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Target number for \(capability.label)")
            } else {
                Picker("Target state", selection: $toggleValue) {
                    Text("Choose on or off").tag("")
                    Text("On").tag("true")
                    Text("Off").tag("false")
                }
                .accessibilityLabel("Target state for \(capability.label)")
            }
            Text("Task: \(resolvedTaskDescription.isEmpty ? "Choose a typed value" : resolvedTaskDescription)")
                .font(.system(size: 12, weight: .medium))
                .textSelection(.enabled)
            Text("The control and typed value define the operation. Sage will recheck the current application and request fresh approval before changing it.")
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func previewPanel(_ preview: GoalPlanPreview) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Proposed procedure · \(preview.steps.count) step(s) · \(preview.waves.count) wave(s)", systemImage: "list.number")
                .font(.system(size: 13, weight: .semibold))
            Text("Estimated critical path: about \(String(format: "%.1f", Double(preview.estimatedElapsedMicros) / 1_000)) ms. This is a local scheduling estimate, not a completion promise.")
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
            ForEach(preview.waves.indices, id: \.self) { waveIndex in
                let labels = preview.waves[waveIndex].compactMap { nodeID in
                    preview.steps.first(where: { $0.id == nodeID })?.capabilityLabel
                }
                Text("Wave \(waveIndex + 1): \(labels.joined(separator: ", "))")
                    .font(.system(size: 11.5, weight: .medium))
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(Array(preview.steps.enumerated()), id: \.element.id) { index, step in
                        VStack(alignment: .leading, spacing: 5) {
                            Text("Step \(index + 1): \(step.capabilityLabel)")
                                .font(.system(size: 12.5, weight: .medium))
                            Text("Input: \(step.inputSummary)")
                                .font(.system(size: 11.5))
                                .foregroundStyle(.secondary)
                            Text("Effect: \(step.effects.joined(separator: ", "))")
                                .font(.system(size: 11.5))
                                .foregroundStyle(.secondary)
                            Text("Verification: \(step.verification)")
                                .font(.system(size: 11.5))
                                .foregroundStyle(.secondary)
                            Text("Restoration: \(step.restoration)")
                                .font(.system(size: 11.5))
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 9))
                    }
                }
            }
            .frame(maxHeight: 150)
            Text("This preview grants no authority. Starting creates a task; every action still needs fresh policy approval and independent verification.")
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(SageTheme.inputFill, in: RoundedRectangle(cornerRadius: 11))
        .overlay(RoundedRectangle(cornerRadius: 11).stroke(SageTheme.stroke))
    }

    private var retryPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Submission result needs confirmation", systemImage: "arrow.triangle.2.circlepath")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(SageTheme.warning)
            Text("Retry resends the same goal with the same submission identity. Sage deduplicates accepted retries so a lost response cannot create another task.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(SageTheme.warning.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func previewSelectedGoal() {
        guard let capability = selectedCapability else { return }
        if capability.input.valueType == "number" {
            guard let number = Double(numberValue.trimmingCharacters(in: .whitespacesAndNewlines)), number.isFinite else { return }
            model.previewGoal(capability: capability, value: .number(number))
        } else if let value = Bool(toggleValue) {
            model.previewGoal(capability: capability, value: .boolean(value))
        }
    }
}
