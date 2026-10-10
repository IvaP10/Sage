import AppKit
import SwiftUI

@main
struct SageApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = AppModel()
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        WindowGroup("Sage", id: "main") {
            MainView(model: model)
                .frame(minWidth: 920, minHeight: 640)
                .preferredColorScheme(.dark)
                .task {
                    appDelegate.model = model
                    await model.start()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1_120, height: 760)
        .commands {
            CommandGroup(after: .newItem) {
                Button("New chat") {
                    model.newTask()
                }
                .keyboardShortcut("n", modifiers: [.command])
                Button("Voice input") {
                    model.toggleVoiceInput()
                }
                .keyboardShortcut(" ", modifiers: [.command, .shift])
            }
        }

        MenuBarExtra {
            Button("Open Sage") {
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
            Button("New chat") {
                NSApplication.shared.activate(ignoringOtherApps: true)
                model.newTask()
            }
            Button(model.worldModelBusy ? "Discovering current app…" : "Discover current app") {
                model.discoverCurrentApplication()
            }
            .disabled(model.worldModelBusy || model.learningSessionID != nil)
            Button("Browser discovery (awaiting live acceptance)") {
                model.discoverPairedBrowser()
            }
            .disabled(true)
            .help("Browser discovery stays unavailable until the exact paired-tab flow passes live Chrome acceptance.")
            Button(model.worldModelBusy ? "Discovering local renderers…" : "Discover local media renderers") {
                model.discoverLocalMediaRenderers()
            }
            .disabled(model.worldModelBusy || model.learningSessionID != nil)
            .help("A passive private-LAN scan reads UPnP descriptions only. Devices remain untrusted and receive no control commands.")
            if !model.upnpRendererCandidates.isEmpty {
                Section("Unpaired renderer candidates") {
                    Text("Passive discovery only. These devices are not trusted and cannot be controlled.")
                    ForEach(model.upnpRendererCandidates) { candidate in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(candidate.friendlyName)
                            Text("\(candidate.manufacturer ?? "Unknown maker") · \(candidate.modelName ?? candidate.deviceType) · \(candidate.source)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if let state = model.upnpRendererTransportStates[candidate.id] {
                                Text("Reported playback: \(state)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            if let formats = model.upnpRendererProtocolSummaries[candidate.id] {
                                Text("Advertised receiver formats: \(formats)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Button("Read playback state") {
                                model.observeRendererTransport(candidate)
                            }
                            .disabled(model.worldModelBusy || model.learningSessionID != nil)
                            .help("Revalidates the discovered renderer and requests read-only AVTransport state. The response remains untrusted.")
                            if candidate.connectionManagerServiceType != nil {
                                Button("Read receiver formats") {
                                    model.observeRendererProtocolInfo(candidate)
                                }
                                .disabled(model.worldModelBusy || model.learningSessionID != nil)
                                .help("Revalidates the renderer and reads its advertised ConnectionManager formats. This does not pair, stream, or control playback.")
                            }
                        }
                    }
                }
            }
            if !model.controllerDrafts.isEmpty {
                Section("Controller drafts") {
                    Text("Review only checks the saved steps against the current app. Run once starts a task with fresh approval for each effect.")
                    ForEach(model.controllerDrafts) { draft in
                        Button("\(draft.systemLabel) · \(draft.stepCount) steps · \(draft.status)") {
                            openWindow(id: "main")
                            NSApplication.shared.activate(ignoringOtherApps: true)
                            model.inspectControllerDraft(draft)
                        }
                        .disabled(model.worldModelBusy || model.pendingDecision != nil)
                    }
                }
            }
            if !model.learningCandidates.isEmpty, model.learningSessionID == nil {
                Section("Learning inbox") {
                    Text("Sage found controls that may support a reversible test in \(model.learningCandidates[0].systemLabel). Nothing has been activated.")
                    Text("Approval covers only the named control, for up to 20 probes or 10 minutes. Each probe must restore and verify its original value.")
                    ForEach(model.learningCandidates) { candidate in
                        Button("Approve reversible test: \(candidate.label)") {
                            model.approveLearningCandidate(candidate)
                        }
                        .disabled(model.worldModelBusy)
                    }
                }
            }
            if let candidate = model.learningApprovedCandidate,
               model.learningSessionID != nil {
                Section("Approved learning session") {
                    Text("\(candidate.systemLabel) • \(candidate.label) (\(candidate.role))")
                    if let expiresAt = model.learningExpiresAt {
                        Text("Expires \(expiresAt, style: .relative). Stop, lock, or adapter disconnect ends this session.")
                    } else {
                        Text("This approval is limited to the named control and will expire within 10 minutes. Stop, lock, or adapter disconnect ends it.")
                    }
                    Button(model.worldModelBusy ? "Probe settling…" : "Run one reversible test") {
                        model.runApprovedLearningProbe()
                    }
                    .disabled(model.worldModelBusy)
                    Button("Stop and revoke learning approval") {
                        model.stopLearningSession()
                    }
                    .disabled(model.worldModelBusy)
                }
            }
            if !model.worldModelStatus.isEmpty {
                Text(model.worldModelStatus)
                    .lineLimit(3)
            }
            Divider()
            Button("Quit Sage") {
                NSApplication.shared.terminate(nil)
            }
        } label: {
            SageMenuBarIcon(hasLearningRequests: !model.learningCandidates.isEmpty && model.learningSessionID == nil)
                .accessibilityLabel(model.learningCandidates.isEmpty ? "Sage" : "Sage, \(model.learningCandidates.count) learning requests")
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationWillTerminate(_ notification: Notification) {
        model?.stop()
    }
}
