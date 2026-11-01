; Build with Inno Setup 6 after producing dist\SZFD.exe.
#ifndef AppVersion
  #define AppVersion "0.6.1"
#endif
[Setup]
AppId={{47BE8653-DFA0-43EA-AD07-6AF5A9F31DCA}
AppName=SZFD
AppVersion={#AppVersion}
AppPublisher=Stalzone Frag Detector
DefaultDirName={localappdata}\Programs\SZFD
DefaultGroupName=SZFD
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=SZFD-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\SZFD.exe

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\SZFD.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\SZFD-CLI.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SZFD"; Filename: "{app}\SZFD.exe"
Name: "{autodesktop}\SZFD"; Filename: "{app}\SZFD.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SZFD.exe"; Description: "{cm:LaunchProgram,SZFD}"; Flags: nowait postinstall skipifsilent
