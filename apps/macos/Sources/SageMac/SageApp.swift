import AppKit
import SwiftUI

@main
struct SageApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup("Sage") {
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
            Divider()
            Button("Quit Sage") {
                NSApplication.shared.terminate(nil)
            }
        } label: {
            SageMenuBarIcon()
                .accessibilityLabel("Sage")
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
