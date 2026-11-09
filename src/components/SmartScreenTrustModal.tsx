import { useState } from "react";
import { Button, Card, CardTitle } from "../ui";

interface SmartScreenTrustModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SmartScreenTrustModal({ isOpen, onClose }: SmartScreenTrustModalProps) {
  const [copiedCmd, setCopiedCmd] = useState(false);

  if (!isOpen) return null;

  const verifyCmd = `Get-FileHash "$HOME\\Downloads\\LocalLLmPanel-setup.exe" -Algorithm SHA256`;

  const handleCopy = () => {
    void navigator.clipboard?.writeText(verifyCmd);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 overflow-y-auto">
      <Card className="w-full max-w-xl p-5 space-y-4 shadow-2xl border-indigo-500/30">
        <div className="flex items-center justify-between border-b border-edge pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xl select-none">🛡️</span>
            <CardTitle>Windows Trust & SmartScreen Guidance</CardTitle>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-lg font-bold"
            title="Close"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4 text-xs leading-relaxed text-slate-300">
          {/* SmartScreen Warning Explanation */}
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3.5 space-y-2">
            <div className="font-semibold text-amber-200 flex items-center gap-2 text-sm">
              <span>⚠️</span>
              <span>Seeing "Windows protected your PC" (SmartScreen)?</span>
            </div>
            <p className="text-amber-100/90 leading-relaxed">
              When installing or updating Local LLM Panel, Windows Defender SmartScreen may display a warning stating
              that Microsoft Defender prevented an unrecognized app from starting.
            </p>
            <div className="text-[11px] text-amber-200/80">
              <strong>Why this happens:</strong> Code-signing certificates for open-source desktop software require
              commercial Extended Validation (EV) certificates that cost hundreds of dollars per year. As a community
              open-source project, our releases are built transparently via GitHub Actions and released directly.
            </div>
          </div>

          {/* How to proceed */}
          <div className="space-y-2">
            <div className="font-semibold text-slate-100 text-sm">How to proceed safely on first run:</div>
            <ol className="list-decimal pl-5 space-y-1.5 text-slate-300">
              <li>
                On the blue SmartScreen dialog, click <strong className="text-indigo-300">"More info"</strong>.
              </li>
              <li>
                Click the <strong className="text-indigo-300">"Run anyway"</strong> button that appears.
              </li>
              <li>
                The installer will start normally. You only need to do this once per release.
              </li>
            </ol>
          </div>

          {/* Cryptographic SHA-256 Checksum Verification */}
          <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-3.5 space-y-2.5">
            <div className="font-semibold text-indigo-200 flex items-center gap-2 text-sm">
              <span>🔒</span>
              <span>Verify Cryptographic SHA-256 Checksum</span>
            </div>
            <p className="text-indigo-100/90 leading-relaxed">
              To verify that your downloaded binary has not been tampered with and exactly matches the official GitHub release,
              run the following command in PowerShell:
            </p>

            <div className="flex items-center justify-between gap-2 bg-black/50 border border-white/10 rounded-md p-2 font-mono text-[11px] text-indigo-300">
              <span className="truncate">{verifyCmd}</span>
              <Button
                variant="ghost"
                className="text-[11px] shrink-0 h-6 px-2 text-slate-200 hover:text-white"
                onClick={handleCopy}
              >
                {copiedCmd ? "✓ Copied" : "Copy"}
              </Button>
            </div>

            <p className="text-[11px] text-indigo-200/80 leading-relaxed">
              Compare the printed 64-character hash against the <code className="bg-black/30 px-1 py-0.5 rounded text-indigo-200">SHA256SUMS.txt</code> file
              published on the official GitHub Release page. If the hashes match, the binary is authentic.
            </p>
          </div>
        </div>

        <div className="flex justify-end pt-2 border-t border-edge">
          <Button variant="primary" onClick={onClose}>
            Got it, close
          </Button>
        </div>
      </Card>
    </div>
  );
}
