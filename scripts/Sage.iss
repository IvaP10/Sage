#define MyAppName "Sage"
#define MyAppVersion "1.0.1-preview.1"
#ifndef SourceDir
  #define SourceDir "dist\windows\win-x64"
#endif
#ifndef OutputDir
  #define OutputDir "dist\windows"
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename "Sage-1.0.1-windows-x64-preview"
#endif

[Setup]
AppId={{A4A1D7A5-12A4-4EAF-94C8-9C4EE8D4F7B7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=IvaP10
DefaultDirName={localappdata}\Sage
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayName={#MyAppName} preview
WizardStyle=modern

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Sage"; Filename: "{app}\Sage.Windows.exe"
Name: "{autodesktop}\Sage"; Filename: "{app}\Sage.Windows.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\Sage.Windows.exe"; Description: "Launch Sage"; Flags: nowait postinstall skipifsilent
