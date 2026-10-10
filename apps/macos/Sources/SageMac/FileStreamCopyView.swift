import AppKit
import SwiftUI

struct FileStreamCopyView: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var sourcePath = ""
    @State private var destinationPath = ""
    @State private var overwrite = false

    private var canStart: Bool {
        !model.worldModelBusy
            && !model.fileStreamCopyNeedsRetry
            && !sourcePath.isEmpty
            && !destinationPath.isEmpty
            && sourcePath != destinationPath
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                Label("Copy a local file", systemImage: "arrow.left.arrow.right")
                    .font(.system(size: 18, weight: .semibold))
                Text("Sage streams one file of up to 16 MiB on this Mac and verifies the destination after writing it.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(alignment: .leading, spacing: 12) {
                pathRow(title: "Source", path: sourcePath, button: "Choose source…", action: chooseSource)
                pathRow(title: "Destination", path: destinationPath, button: "Choose destination…", action: chooseDestination)
                Toggle("Replace the destination if it already exists", isOn: $overwrite)
                    .toggleStyle(.checkbox)
                    .font(.system(size: 12))
                    .disabled(model.worldModelBusy || model.fileStreamCopyNeedsRetry)
            }
            .padding(14)
            .background(SageTheme.inputFill, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(SageTheme.stroke))

            Text("Sage will show separate approvals for reading the source and writing the destination. The request grants no ongoing access.")
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if !model.fileStreamCopyStatus.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    if model.worldModelBusy { ProgressView().controlSize(.small) }
                    Text(model.fileStreamCopyStatus)
                        .font(.system(size: 12))
                        .foregroundStyle(model.fileStreamCopyNeedsRetry ? SageTheme.warning : Color.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Divider()
            HStack {
                Button("Later") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .disabled(model.worldModelBusy)
                Spacer()
                if model.fileStreamCopyNeedsRetry {
                    Button("Retry exact request") { model.retryFileStreamCopy() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(model.worldModelBusy)
                } else {
                    Button("Start and request approvals") {
                        model.submitFileStreamCopy(
                            sourcePath: sourcePath,
                            destinationPath: destinationPath,
                            overwrite: overwrite
                        )
                    }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canStart)
                }
            }
        }
        .padding(24)
        .frame(minWidth: 520, idealWidth: 620, maxWidth: 720, minHeight: 390, idealHeight: 440)
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled(model.worldModelBusy)
    }

    private func pathRow(title: String, path: String, button: String, action: @escaping () -> Void) -> some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                Text(path.isEmpty ? "Not selected" : path)
                    .font(.system(size: 11))
                    .foregroundStyle(path.isEmpty ? Color.secondary : Color.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .accessibilityLabel(path.isEmpty ? "No \(title.lowercased()) selected" : "\(title): \(path)")
            }
            Spacer(minLength: 8)
            Button(button, action: action)
                .buttonStyle(SageBorderedButtonStyle())
                .disabled(model.worldModelBusy || model.fileStreamCopyNeedsRetry)
        }
    }

    private func chooseSource() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose source"
        panel.message = "Choose the exact local file Sage should read."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        sourcePath = url.path
        if destinationPath.isEmpty {
            destinationPath = url.deletingLastPathComponent()
                .appendingPathComponent("Copy of \(url.lastPathComponent)")
                .path
        }
    }

    private func chooseDestination() {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.prompt = "Choose destination"
        panel.message = "Choose where Sage should write the verified copy."
        if !sourcePath.isEmpty {
            let source = URL(fileURLWithPath: sourcePath)
            panel.directoryURL = source.deletingLastPathComponent()
            panel.nameFieldStringValue = "Copy of \(source.lastPathComponent)"
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        destinationPath = url.path
    }
}
