import SwiftUI

struct MemorySettingsView: View {
    @Bindable var model: AppModel
    @State private var search = ""
    @State private var newMemory = ""
    @State private var editing: MemoryRecord?
    @State private var editText = ""
    @State private var deleting: MemoryRecord?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Memory").font(.headline)
                Spacer()
                Toggle("Use memory", isOn: Binding(get: { model.memoryEnabled }, set: { model.knowledge("configure", enabled: $0) }))
            }
            TextField("Search memories", text: $search).textFieldStyle(.roundedBorder)
            ForEach(model.memories.filter { search.isEmpty || ($0.subject + " " + $0.content).localizedCaseInsensitiveContains(search) }) { memory in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(memory.subject).fontWeight(.medium)
                        Text(memory.kind).foregroundStyle(.secondary).font(.caption)
                        Spacer()
                        Toggle("Enabled", isOn: Binding(get: { memory.enabled }, set: { model.knowledge($0 ? "enable" : "disable", id: memory.id) })).labelsHidden().help("Use this memory")
                        Button("Edit") { editing = memory; editText = memory.content }.disabled(!model.memoryEnabled)
                        Button("Delete", role: .destructive) { deleting = memory }
                    }
                    Text(memory.content).foregroundStyle(.secondary).textSelection(.enabled)
                }.padding(.vertical, 6)
                Divider()
            }
            if model.memories.isEmpty { Text("No memories saved.").foregroundStyle(.secondary) }
            HStack {
                TextField("Remember that…", text: $newMemory).textFieldStyle(.roundedBorder)
                Button("Remember") { model.knowledge("remember", content: newMemory); newMemory = "" }
                    .disabled(!model.memoryEnabled || newMemory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 14))
        .onAppear { model.knowledge("list") }
        .sheet(item: $editing) { memory in
            VStack(alignment: .leading, spacing: 16) {
                Text("Edit memory").font(.headline)
                TextEditor(text: $editText).frame(width: 430, height: 120)
                HStack { Button("Cancel") { editing = nil }; Spacer(); Button("Save") { model.knowledge("edit", id: memory.id, content: editText); editing = nil }.disabled(editText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) }
            }.padding(24)
        }
        .alert("Delete this memory?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            Button("Delete", role: .destructive) { if let memory = deleting { model.knowledge("delete", id: memory.id) }; deleting = nil }
            Button("Cancel", role: .cancel) { deleting = nil }
        }
    }
}

struct WorkflowSettingsView: View {
    @Bindable var model: AppModel
    @State private var name = ""
    @State private var request = ""
    @State private var date = Date().addingTimeInterval(3600)
    @State private var recurrence = 0
    @State private var folder = ""
    @State private var workflowName = ""
    @State private var selectedSkills: [String] = []
    @State private var reviewing: SkillRecord?
    @State private var forgetting: RoutineRecord?
    @State private var prefixSkillNames: [String: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Skills and background tasks").font(.headline)
            VStack(alignment: .leading, spacing: 10) {
                Toggle("Learn and reuse routines", isOn: Binding(
                    get: { model.routineLearningEnabled },
                    set: { enabled in model.workflow("configure_learning", json: "{\"enabled\":\(enabled)}") }
                )).disabled(!model.memoryEnabled)
                Text(model.memoryEnabled
                    ? "Sage looks for repeated steps on this device. After three verified runs, review a routine before Sage can use it. Each run checks folder access again."
                    : "Turn on Memory to learn routines. Each run will still check folder access again.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.routineLearningEnabled && model.routines.isEmpty {
                    Text("No repeated routines yet. Keep using Sage and verified patterns will appear here.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(model.routines) { routine in
                    VStack(alignment: .leading, spacing: 7) {
                        Text(routine.requests.first ?? "Repeated routine").font(.system(size: 13, weight: .medium))
                            .textSelection(.enabled).lineLimit(2)
                        Text("Observed in \(routine.verifiedRuns) verified tasks")
                            .font(.caption).foregroundStyle(.secondary)
                        if let evolution = routine.evolution {
                            let unchanged = evolution.unchangedEffectClasses.isEmpty
                                ? "none"
                                : evolution.unchangedEffectClasses.joined(separator: ", ")
                            let added = evolution.addedEffectClasses.isEmpty
                                ? "none"
                                : evolution.addedEffectClasses.joined(separator: ", ")
                            let removed = evolution.removedEffectClasses.isEmpty
                                ? "none"
                                : evolution.removedEffectClasses.joined(separator: ", ")
                            Text("Corrected variation of ‘\(evolution.sourceRequest)’ (\(evolution.sourceVerifiedRuns) source runs).")
                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            Text("Effect classes — same: \(unchanged); added: \(added); removed: \(removed). This stays a draft until three independent runs and review.")
                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        if !routine.ready {
                            Text("Review opens after three verified tasks.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else if !model.routineLearningEnabled {
                            Text("Turn on routine learning to review and use this routine.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        ForEach(Array(routine.steps.enumerated()), id: \.offset) { index, step in
                            Text("\(index + 1). \(step)").font(.caption).textSelection(.enabled)
                        }
                        if !routine.enabled {
                            Button("Review and enable") {
                                let body = try? JSONSerialization.data(withJSONObject: ["digest": routine.reviewDigest])
                                model.workflow("review_routine", id: routine.id, json: body.map { String(decoding: $0, as: UTF8.self) } ?? "")
                            }.disabled(!routine.ready || !model.routineLearningEnabled)
                        } else {
                            Label("Reviewed and ready", systemImage: "checkmark.circle.fill")
                                .font(.caption).foregroundStyle(SageTheme.accent)
                        }
                        Button("Forget this routine", role: .destructive) {
                            forgetting = routine
                        }
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                        .background(SageTheme.canvas, in: RoundedRectangle(cornerRadius: 10))
                }
                ForEach(model.routineFamilies) { family in
                    VStack(alignment: .leading, spacing: 8) {
                        Label("\(family.branches.count) reviewed routines share these steps", systemImage: "arrow.triangle.branch")
                            .font(.system(size: 13, weight: .medium))
                        Text("Sage found the same verified steps across these routines. Different next steps stay separate from the skill draft.")
                            .font(.caption).foregroundStyle(.secondary)
                        Label("Common steps", systemImage: "checkmark.circle")
                            .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        ForEach(Array(family.sharedSteps.enumerated()), id: \.offset) { index, step in
                            Text("\(index + 1). \(step)").font(.caption).textSelection(.enabled)
                        }
                        Label("Different next steps", systemImage: "arrow.turn.down.right")
                            .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        ForEach(family.branches) { branch in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text("When you ask")
                                        .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                                    Spacer()
                                    Text("\(branch.verifiedRuns) verified runs")
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                                Text(branch.requests.joined(separator: " · "))
                                    .font(.caption).textSelection(.enabled)
                                if branch.nextSteps.isEmpty {
                                    Text("No additional steps in this branch.").font(.caption).foregroundStyle(.secondary)
                                } else {
                                    ForEach(Array(branch.nextSteps.enumerated()), id: \.offset) { stepIndex, step in
                                        Text("Then \(stepIndex + 1). \(step)").font(.caption).textSelection(.enabled)
                                    }
                                }
                            }
                            .padding(9).frame(maxWidth: .infinity, alignment: .leading)
                            .background(SageTheme.canvas, in: RoundedRectangle(cornerRadius: 8))
                        }
                        TextField("Name this shared skill", text: Binding(
                            get: { prefixSkillNames[family.id] ?? "" },
                            set: { prefixSkillNames[family.id] = $0 }
                        )).textFieldStyle(.roundedBorder)
                        Button("Create draft for review") {
                            let body = try? JSONSerialization.data(withJSONObject: ["digest": family.reviewDigest])
                            model.workflow(
                                "synthesize_skill",
                                id: family.id,
                                name: prefixSkillNames[family.id] ?? "",
                                json: body.map { String(decoding: $0, as: UTF8.self) } ?? ""
                            )
                            prefixSkillNames[family.id] = ""
                        }
                        .disabled((prefixSkillNames[family.id] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        Label("Review the draft before enabling it. Each run checks access again.", systemImage: "hand.raised")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                    .background(SageTheme.canvas, in: RoundedRectangle(cornerRadius: 10))
                }
            }.padding(14).background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 12))
            ForEach(model.skills) { skill in
                HStack {
                    Toggle(skill.name, isOn: Binding(get: { selectedSkills.contains(skill.id) }, set: { enabled in
                        if enabled { selectedSkills.append(skill.id) } else { selectedSkills.removeAll { $0 == skill.id } }
                    })).disabled(!skill.enabled)
                    Spacer()
                    if skill.enabled { Button("Run") { model.workflow("run_skill", id: skill.id) } }
                    else if skill.sourcePaused {
                        Label("Paused: review source routines", systemImage: "pause.circle")
                            .font(.caption).foregroundStyle(.secondary)
                    } else { Button("Review draft") { reviewing = skill } }
                    Button("Delete", role: .destructive) { model.workflow("delete_skill", id: skill.id) }
                }
            }
            if !model.skills.isEmpty {
                HStack {
                    TextField("Workflow name", text: $workflowName).textFieldStyle(.roundedBorder)
                    Button("Create workflow") {
                        let value: [String: Any] = ["id": UUID().uuidString.lowercased(), "name": workflowName, "skill_ids": selectedSkills, "enabled": true]
                        if let data = try? JSONSerialization.data(withJSONObject: value) { model.workflow("save_workflow", json: String(decoding: data, as: UTF8.self)); workflowName = ""; selectedSkills = [] }
                    }.disabled(selectedSkills.isEmpty || workflowName.isEmpty)
                }
            }
            ForEach(model.workflows) { workflow in
                HStack {
                    Text(workflow.name)
                    if workflow.sourcePaused { Text("Paused: review source routines").font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                    Button("Run") { model.workflow("run_workflow", id: workflow.id) }.disabled(!workflow.enabled)
                    Button("Delete", role: .destructive) { model.workflow("delete_workflow", id: workflow.id) }
                }
            }
            Divider()
            ForEach(model.schedules) { schedule in
                VStack(alignment: .leading) {
                    HStack { Text(schedule.name); Text(schedule.enabled ? "Scheduled" : "Inactive").foregroundStyle(.secondary); Spacer(); Button("Delete", role: .destructive) { model.workflow("delete_schedule", id: schedule.id) } }
                    Text(schedule.request).foregroundStyle(.secondary)
                    if let error = schedule.lastError { Text(error).foregroundStyle(.red) }
                    if schedule.sourcePaused { Text("Paused: review source routines before scheduling again.").font(.caption).foregroundStyle(.secondary) }
                }
            }
            TextField("Task name", text: $name).textFieldStyle(.roundedBorder)
            TextField("What should happen?", text: $request).textFieldStyle(.roundedBorder)
            DatePicker("Start", selection: $date)
            Picker("Repeat", selection: $recurrence) {
                Text("Once").tag(0); Text("Every hour").tag(3600); Text("Every day").tag(86400)
            }
            TextField("Watch a folder (optional)", text: $folder).textFieldStyle(.roundedBorder)
            Text("Expires after 30 days or 100 runs. A watched folder may be read in the background. Additional access pauses for your approval.").font(.caption).foregroundStyle(.secondary)
            Button("Schedule") { model.schedule(name: name, request: request, date: date, interval: recurrence, folder: folder); name = ""; request = ""; folder = "" }
                .disabled(name.isEmpty || request.isEmpty)
        }
        .padding(20)
        .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 14))
        .onAppear { model.workflow("list") }
        .sheet(item: $reviewing) { skill in
            VStack(alignment: .leading, spacing: 16) {
                Text(skill.name).font(.headline)
                Text("Review the saved steps. Each run still needs permission for its resources and effects.")
                ScrollView { Text(skill.preview).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(width: 520, height: 320)
                HStack {
                    Button("Cancel") { reviewing = nil }; Spacer()
                    Button("Enable skill") {
                        if let data = try? JSONSerialization.data(withJSONObject: ["digest": skill.reviewDigestCandidate]) {
                            model.workflow("review_skill", id: skill.id, json: String(decoding: data, as: UTF8.self))
                        }
                        reviewing = nil
                    }
                }
            }.padding(24)
        }
        .alert("Forget this routine?", isPresented: Binding(
            get: { forgetting != nil },
            set: { if !$0 { forgetting = nil } }
        )) {
            Button("Forget routine", role: .destructive) {
                if let routine = forgetting { model.workflow("forget_routine", id: routine.id) }
                forgetting = nil
            }
            Button("Cancel", role: .cancel) { forgetting = nil }
        } message: {
            Text("This removes the observed examples and your review from this device.")
        }
    }
}
