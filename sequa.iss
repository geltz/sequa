[Setup]
AppId={{364BC968-6F38-41C6-8567-5A7836514CB0}
AppName=sequa
AppVersion=2.0
AppPublisher=geltz
DefaultDirName={autopf}\sequa
SetupIconFile=sequa.ico
DefaultGroupName=sequa
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=.
OutputBaseFilename=sequa_setup_2.0
WizardStyle=modern
UninstallDisplayIcon={app}\sequa.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\sequa\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\sequa"; Filename: "{app}\sequa.exe"
Name: "{group}\{cm:UninstallProgram,sequa}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\sequa"; Filename: "{app}\sequa.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\sequa.exe"; Description: "{cm:LaunchProgram,sequa}"; Flags: nowait postinstall skipifsilent