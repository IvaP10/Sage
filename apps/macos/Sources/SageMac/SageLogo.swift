import AppKit
import SwiftUI

@MainActor
struct SageLogo: View {
    let size: CGFloat

    var body: some View {
        Image(nsImage: SageBrand.logo)
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .frame(width: size, height: size)
            .clipShape(RoundedRectangle(cornerRadius: size * 0.2, style: .continuous))
            .accessibilityHidden(true)
    }
}

@MainActor
struct SageMenuBarIcon: View {
    var body: some View {
        Image(nsImage: SageBrand.menuBarIcon)
            .frame(width: 18, height: 18)
            .accessibilityHidden(true)
    }
}

@MainActor
enum SageBrand {
    static let logo = loadLogo()
    static let menuBarIcon = makeMenuBarIcon()

    private static func loadLogo() -> NSImage {
        let bundledLogo = Bundle.main.url(forResource: "sage-logo", withExtension: "png")
        let developmentLogo = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("assets/icon-source-bg.png")
        let legacyDevelopmentLogo = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("assets/icon-source.png")
        let bundledIcon = Bundle.main.url(forResource: "sage", withExtension: "icns")

        for url in [bundledLogo, developmentLogo, legacyDevelopmentLogo, bundledIcon].compactMap({ $0 }) {
            if let image = NSImage(contentsOf: url) {
                return image
            }
        }

        return NSApplication.shared.applicationIconImage
    }

    /// MenuBarExtra must receive a small template image with an intrinsic menu-bar
    /// size. Passing the full 1024-pixel application artwork can make macOS lay out
    /// one of its image representations at full scale and clip it into the menu bar.
    private static func makeMenuBarIcon() -> NSImage {
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            let mark = NSBezierPath()
            mark.move(to: NSPoint(x: rect.maxX - 4, y: rect.maxY - 3))
            mark.line(to: NSPoint(x: rect.minX + 4, y: rect.midY + 3))
            mark.line(to: NSPoint(x: rect.maxX - 4, y: rect.midY - 2))
            mark.line(to: NSPoint(x: rect.minX + 4, y: rect.minY + 3))
            mark.lineWidth = 3.4
            mark.lineCapStyle = .round
            mark.lineJoinStyle = .round
            NSColor.black.setStroke()
            mark.stroke()
            return true
        }
        image.isTemplate = true
        return image
    }
}
