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

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Skills and background tasks").font(.headline)
            ForEach(model.skills) { skill in
                HStack {
                    Toggle(skill.name, isOn: Binding(get: { selectedSkills.contains(skill.id) }, set: { enabled in
                        if enabled { selectedSkills.append(skill.id) } else { selectedSkills.removeAll { $0 == skill.id } }
                    })).disabled(!skill.enabled)
                    Spacer()
                    if skill.enabled { Button("Run") { model.workflow("run_skill", id: skill.id) } }
                    else { Button("Review draft") { reviewing = skill } }
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
                HStack { Text(workflow.name); Spacer(); Button("Run") { model.workflow("run_workflow", id: workflow.id) }; Button("Delete", role: .destructive) { model.workflow("delete_workflow", id: workflow.id) } }
            }
            Divider()
            ForEach(model.schedules) { schedule in
                VStack(alignment: .leading) {
                    HStack { Text(schedule.name); Text(schedule.enabled ? "Scheduled" : "Inactive").foregroundStyle(.secondary); Spacer(); Button("Delete", role: .destructive) { model.workflow("delete_schedule", id: schedule.id) } }
                    Text(schedule.request).foregroundStyle(.secondary)
                    if let error = schedule.lastError { Text(error).foregroundStyle(.red) }
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
    }
}
