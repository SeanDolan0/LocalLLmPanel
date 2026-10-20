#!/usr/bin/env python3
"""
Local LLM Panel - Release Manager TUI
Interactive release tool with a modern two-pane developer dashboard.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

REPO_ROOT = Path(r"C:\Users\sedol\Documents\LocalLLmPanel")

VERSION_FILES = [
    {
        "path": "package.json",
        "pattern": r'"version"\s*:\s*"[\d\.\-a-zA-Z]+"',
        "replacement": '"version": "',
        "count": 1,
        "flags": 0,
        "kind": "json",
    },
    {
        "path": "package-lock.json",
        "pattern": r'"version"\s*:\s*"[\d\.\-a-zA-Z]+"',
        "replacement": '"version": "',
        "count": 2,
        "flags": 0,
        "kind": "json",
    },
    {"path": "src-tauri/Cargo.toml", "kind": "cargo_toml"},
    {
        "path": "src-tauri/tauri.conf.json",
        "pattern": r'"version"\s*:\s*"[\d\.\-a-zA-Z]+"',
        "replacement": '"version": "',
        "count": 1,
        "flags": 0,
        "kind": "json",
    },
]

_CARGO_VERSION_LINE = re.compile(r'^version\s*=\s*"([\d\.\-a-zA-Z]+)"', re.MULTILINE)


def _cargo_package_span(content: str) -> tuple[int, int]:
    header = re.search(r"^\[package\]\s*$", content, re.MULTILINE)
    if not header:
        raise Exception("Cargo.toml: [package] table not found")
    start = header.end()
    next_table = re.search(r"^\[", content[start:], re.MULTILINE)
    end = start + next_table.start() if next_table else len(content)
    return start, end


BUILD_TYPES = [
    {"id": "setup", "label": "Standard Release", "desc": "NSIS installer (~40s)"},
    {
        "id": "all",
        "label": "Full Distribution",
        "desc": "NSIS + MSI installers (~2-3m)",
    },
]


def _find_llvm_bin() -> Optional[str]:
    if sys.platform != "win32":
        return None
    for p in (r"C:\Program Files\LLVM\bin", r"C:\Program Files (x86)\LLVM\bin"):
        if os.path.exists(os.path.join(p, "lld-link.exe")):
            return p
    return None


LLVM_BIN = _find_llvm_bin()

STEPS = [
    {
        "name": "Version Bump",
        "fn": "version_bump",
        "desc": "Sync versions across 4 manifests",
    },
    {
        "name": "Preflight Checks",
        "fn": "preflight",
        "desc": "TypeScript compile & Cargo tests",
    },
    {
        "name": "Build Artifacts",
        "fn": "build",
        "desc": "Bundle Tauri binaries & installers",
    },
    {
        "name": "Git Commit & Tag",
        "fn": "git",
        "desc": "Stage manifests and generate release tag",
    },
    {
        "name": "GitHub Release",
        "fn": "release",
        "desc": "Push commits & create gh release",
    },
]


class DirtyRepoError(Exception):
    pass


@dataclass
class ReleaseConfig:
    version: str = ""
    build_type: str = "setup"
    skip_tests: bool = False
    allow_dirty: bool = False
    git_branch: str = "main"


def bump_semver(version: str, bump_type: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", version.strip())
    if not m:
        return version
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "major":
        return f"{major + 1}.0.0"
    return version


# -------------------------------------------------------------------------
# Step Row Component
# -------------------------------------------------------------------------


class StepWidget(Static):
    """Refined pipeline step item with badge indicators and detail strings."""

    def __init__(self, step_name: str, desc: str, index: int):
        super().__init__()
        self.step_name = step_name
        self.desc = desc
        self.index = index
        self.status = "pending"
        self.detail = ""
        self.elapsed = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(classes="step-row", id=f"step-row-{self.index}"):
            yield Static(f"{self.index:02d}", classes="step-num")
            with Vertical(classes="step-text"):
                yield Static(self.step_name, classes="step-title")
                yield Static(
                    self.desc, classes="step-desc", id=f"step-desc-{self.index}"
                )
            yield Static(
                "READY",
                classes="step-badge badge-pending",
                id=f"step-badge-{self.index}",
            )

    def set_status(self, status: str, detail: str = "", elapsed: float = 0.0) -> None:
        self.status = status
        self.detail = detail
        self.elapsed = elapsed

        badge = self.query_one(f"#step-badge-{self.index}", Static)
        desc_label = self.query_one(f"#step-desc-{self.index}", Static)
        row = self.query_one(f"#step-row-{self.index}", Horizontal)

        badge.remove_class(
            "badge-pending", "badge-running", "badge-done", "badge-error"
        )
        row.remove_class("step--active", "step--done", "step--error")

        if status == "running":
            badge.update("RUNNING")
            badge.add_class("badge-running")
            row.add_class("step--active")
            desc_label.update("[#f0883e]Executing step tasks...[/#f0883e]")
        elif status == "done":
            time_str = f" ({self.elapsed:.1f}s)" if self.elapsed > 0 else ""
            badge.update(f"DONE{time_str}")
            badge.add_class("badge-done")
            row.add_class("step--done")
            desc_text = self.detail if self.detail else "Completed successfully."
            desc_label.update(f"[#3fb950]✓ {desc_text}[/#3fb950]")
        elif status == "error":
            badge.update("FAILED")
            badge.add_class("badge-error")
            row.add_class("step--error")
            desc_label.update(f"[#f85149]✗ {self.detail}[/#f85149]")
        else:
            badge.update("READY")
            badge.add_class("badge-pending")
            desc_label.update(self.desc)


# -------------------------------------------------------------------------
# Config Screen
# -------------------------------------------------------------------------


class ConfigScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.quit", "Quit", show=True),
        Binding("enter", "submit", "Start", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.config = ReleaseConfig()
        self.current_version = self.get_current_version()
        self.git_branch = self.detect_git_branch()
        self.config.git_branch = self.git_branch

    def get_current_version(self) -> str:
        try:
            pkg_path = REPO_ROOT / "package.json"
            if pkg_path.exists():
                with open(pkg_path) as f:
                    return json.load(f).get("version", "0.0.0")
        except Exception:
            pass
        return "0.0.0"

    def detect_git_branch(self) -> str:
        try:
            r = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return r.stdout.strip() or "main"
        except Exception:
            return "main"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(classes="app-layout-centered"):
            with Vertical(classes="panel-card config-dialog"):
                # Top Header Banner
                with Horizontal(classes="dialog-header"):
                    with Vertical():
                        yield Static(
                            "RELEASE WORKFLOW ORCHESTRATOR",
                            classes="text-accent text-bold",
                        )
                        yield Static(
                            "Configure package manifest updates and binary builds",
                            classes="text-muted",
                        )
                    yield Static(f"branch: {self.git_branch}", classes="branch-tag")

                # Main Form Grid (2 Columns)
                with Horizontal(classes="dialog-body"):
                    # Left Column: Version Management
                    with Vertical(classes="dialog-col"):
                        yield Static("VERSION SELECTION", classes="col-header")

                        with Horizontal(classes="current-ver-card"):
                            yield Static("Active Version:", classes="text-muted")
                            yield Static(
                                f"v{self.current_version}",
                                classes="text-bold text-accent",
                            )

                        yield Label("Target Version bump:", classes="input-caption")
                        with Horizontal(classes="quick-bump-row"):
                            yield Button(
                                f"Patch · {bump_semver(self.current_version, 'patch')}",
                                id="bump-patch",
                                classes="btn-bump",
                            )
                            yield Button(
                                f"Minor · {bump_semver(self.current_version, 'minor')}",
                                id="bump-minor",
                                classes="btn-bump",
                            )
                            yield Button(
                                f"Major · {bump_semver(self.current_version, 'major')}",
                                id="bump-major",
                                classes="btn-bump",
                            )

                        yield Label("Manual Version Override:", classes="input-caption")
                        yield Input(
                            value=self.current_version,
                            placeholder="e.g. 1.2.0",
                            id="version-input",
                        )

                        with Vertical(classes="affected-files-box"):
                            yield Static(
                                "Targets to bump automatically:", classes="mini-header"
                            )
                            for vf in VERSION_FILES:
                                yield Static(
                                    f"• {vf['path']}", classes="text-muted file-entry"
                                )

                    # Right Column: Profile & Parameters
                    with Vertical(classes="dialog-col"):
                        yield Static("BUILD PROFILE", classes="col-header")
                        with RadioSet(id="build-profile-radio"):
                            for idx, bt in enumerate(BUILD_TYPES):
                                yield RadioButton(
                                    f"{bt['label']} [dim]({bt['desc']})[/dim]",
                                    value=(idx == 0),
                                    id=f"bt-{bt['id']}",
                                )

                        yield Static("SAFETY GUARDS", classes="col-header mt-1")
                        yield Checkbox("Skip cargo unit test suite", id="skip-tests")
                        yield Checkbox(
                            "Allow release with other uncommitted files",
                            value=True,
                            id="allow-untracked",
                        )

                # Footer Action Row
                with Horizontal(classes="dialog-actions"):
                    yield Static(
                        "ESC to exit  •  TAB to switch fields", classes="action-hint"
                    )
                    yield Button("Cancel", variant="default", id="cancel-btn")
                    yield Button(
                        "Launch Release Pipeline ›", variant="primary", id="start-btn"
                    )

        yield Footer()

    @on(Button.Pressed, "#bump-patch")
    def on_patch(self) -> None:
        self.query_one("#version-input", Input).value = bump_semver(
            self.current_version, "patch"
        )

    @on(Button.Pressed, "#bump-minor")
    def on_minor(self) -> None:
        self.query_one("#version-input", Input).value = bump_semver(
            self.current_version, "minor"
        )

    @on(Button.Pressed, "#bump-major")
    def on_major(self) -> None:
        self.query_one("#version-input", Input).value = bump_semver(
            self.current_version, "major"
        )

    @on(RadioSet.Changed, "#build-profile-radio")
    def on_profile_change(self, event: RadioSet.Changed) -> None:
        if event.pressed.id == "bt-all":
            self.config.build_type = "all"
        else:
            self.config.build_type = "setup"

    @on(Checkbox.Changed, "#skip-tests")
    def on_skip_tests(self, event: Checkbox.Changed) -> None:
        self.config.skip_tests = event.value

    @on(Checkbox.Changed, "#allow-untracked")
    def on_allow_untracked(self, event: Checkbox.Changed) -> None:
        self.config.allow_dirty = event.value

    @on(Button.Pressed, "#start-btn")
    def action_submit(self) -> None:
        val = self.query_one("#version-input", Input).value.strip()
        if not val:
            self.app.notify("Version cannot be empty", severity="error")
            return
        if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$", val):
            self.app.notify(
                "Invalid semver format (use x.y.z or x.y.z-rc1)", severity="error"
            )
            return

        self.config.version = val
        self.app.push_screen(PipelineScreen(self.config))

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.app.exit()


# -------------------------------------------------------------------------
# Pipeline Screen (Cockpit Dashboard)
# -------------------------------------------------------------------------


class PipelineScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.quit", "Quit"),
    ]

    time_elapsed = reactive(0)

    def __init__(self, config: ReleaseConfig):
        super().__init__()
        self.config = config
        self.step_widgets = []
        self.pending_resume_step = 0
        self.start_timestamp = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(classes="cockpit-root"):
            # Cockpit Header
            with Horizontal(classes="cockpit-topbar"):
                with Horizontal():
                    yield Static(
                        "RELEASE EXECUTION CONSOLE", classes="text-bold text-accent"
                    )
                    yield Static(f"v{self.config.version}", classes="version-badge")
                yield Static(
                    f"Profile: {self.config.build_type.upper()} | LLVM Linker: {'ACTIVE' if LLVM_BIN else 'SYSTEM'}",
                    classes="text-muted",
                )

            # Split Screen View
            with Horizontal(classes="cockpit-body"):
                # Left Pane: Stepper
                with Vertical(classes="cockpit-left-pane"):
                    yield Static("PIPELINE STEPS", classes="col-header")
                    with Vertical(classes="stepper-list"):
                        for i, step in enumerate(STEPS):
                            widget = StepWidget(step["name"], step["desc"], i + 1)
                            self.step_widgets.append(widget)
                            yield widget

                    yield Static("PROGRESS", classes="col-header mt-1")
                    yield ProgressBar(
                        total=len(STEPS), show_eta=False, id="pipeline-progress"
                    )

                    yield Button(
                        "⚡ Force Continue (Bypass Dirty Repo)",
                        variant="warning",
                        id="btn-force-continue",
                        classes="hidden",
                    )

                # Right Pane: Streaming Log Viewer
                with Vertical(classes="cockpit-right-pane"):
                    with Horizontal(classes="log-header-bar"):
                        yield Static("STDOUT & TELEMETRY STREAM", classes="col-header")
                        yield Static("AUTOSCROLL ON", classes="stream-tag")
                    yield RichLog(id="console-log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        self.start_timestamp = time.time()
        self.run_pipeline()

    def write_log(self, message: str, color: str = "") -> None:
        log_view = self.query_one("#console-log", RichLog)
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[#4d5b66]{timestamp}[/#4d5b66] "
        if color:
            log_view.write(f"{prefix}[{color}]{message}[/{color}]")
        else:
            log_view.write(f"{prefix}{message}")

    def write_log_from_worker(self, message: str, color: str = "") -> None:
        self.app.call_from_thread(self.write_log, message, color)

    @on(Button.Pressed, "#btn-force-continue")
    def on_force_continue(self, event: Button.Pressed) -> None:
        event.button.add_class("hidden")
        self.config.allow_dirty = True
        self.write_log("Forcing continue: uncommitted changes bypassed.", "yellow")
        self.run_pipeline(resume_from=self.pending_resume_step)

    @work(exclusive=True, thread=True)
    def run_pipeline(self, resume_from: int = 0) -> None:
        try:
            total_steps = len(STEPS)
            progress_bar = self.query_one("#pipeline-progress", ProgressBar)

            for i in range(resume_from, total_steps):
                step = STEPS[i]
                widget = self.step_widgets[i]
                self.app.call_from_thread(widget.set_status, "running")
                self.write_log_from_worker(
                    f"Starting Step {i+1}: {step['name']}", "cyan"
                )
                step_start = time.time()

                try:
                    method = getattr(self, f"run_{step['fn']}")
                    result = method(self.config.version, self.config.build_type)
                    step_duration = time.time() - step_start

                    self.app.call_from_thread(
                        widget.set_status,
                        "done",
                        result.get("detail", ""),
                        step_duration,
                    )
                    self.write_log_from_worker(
                        f"Step {i+1} completed in {step_duration:.1f}s", "green"
                    )
                    self.app.call_from_thread(progress_bar.update, progress=i + 1)

                except DirtyRepoError as e:
                    self.app.call_from_thread(widget.set_status, "error", str(e))
                    self.write_log_from_worker(
                        f"Uncommitted Changes Detected: {e}", "yellow"
                    )
                    self.pending_resume_step = i
                    self.app.call_from_thread(
                        self.query_one("#btn-force-continue", Button).remove_class,
                        "hidden",
                    )
                    return
                except Exception as e:
                    self.app.call_from_thread(widget.set_status, "error", str(e))
                    self.write_log_from_worker(
                        f"Fatal error at step '{step['name']}': {e}", "red"
                    )
                    return

            total_elapsed = time.time() - self.start_timestamp
            self.write_log_from_worker(
                f"★ Pipeline succeeded in {total_elapsed:.1f}s! Release v{self.config.version} live.",
                "bold green",
            )

        except Exception as e:
            self.write_log_from_worker(f"Unhandled exception: {e}", "bold red")

    def run_command(
        self, args: list[str], cwd: Path | None = None, timeout: int = 300
    ) -> dict:
        cwd = cwd or REPO_ROOT
        try:
            # On Windows, need shell=True to resolve .cmd wrappers (npm.cmd, cargo.cmd, etc.)
            shell = sys.platform == "win32"
            env = os.environ.copy()
            if LLVM_BIN:
                env["PATH"] = f"{LLVM_BIN}{os.pathsep}{env.get('PATH', '')}"

            result = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=shell,
                env=env,
            )
            output = (result.stdout + result.stderr).strip()
            if output:
                for line in output.splitlines()[-8:]:
                    self.write_log_from_worker(f"  {line}", "dim")

            if result.returncode != 0:
                raise Exception(f"Exit {result.returncode}: {output}")
            return {"success": True, "output": output}
        except subprocess.TimeoutExpired:
            raise Exception(f"Timed out after {timeout}s: {' '.join(args)}")

    def run_version_bump(self, version: str, build_type: str) -> dict:
        updated = []
        for f in VERSION_FILES:
            path = REPO_ROOT / f["path"]
            if not path.exists():
                raise Exception(f"Missing file: {path}")

            content = path.read_text(encoding="utf-8")
            if f["kind"] == "cargo_toml":
                start, end = _cargo_package_span(content)
                section, n = _CARGO_VERSION_LINE.subn(
                    f'version = "{version}"', content[start:end], count=1
                )
                if n != 1:
                    raise Exception(f"{f['path']}: [package] version target missing")
                new_content = content[:start] + section + content[end:]
            else:
                new_content, n = re.subn(
                    f["pattern"],
                    f'{f["replacement"]}{version}"',
                    content,
                    count=f["count"],
                    flags=f["flags"],
                )
                if n < f["count"]:
                    raise Exception(
                        f"{f['path']}: expected {f['count']} targets, found {n}"
                    )

            path.write_text(new_content, encoding="utf-8")
            updated.append(Path(f["path"]).name)

        return {"detail": f"Bumped {len(updated)} files ({', '.join(updated)})"}

    def run_preflight(self, version: str, build_type: str) -> dict:
        self.write_log("Running typescript and cargo checks...")
        self.run_command(["npm", "run", "check"])
        if not self.config.skip_tests:
            self.write_log("Running Rust unit test suites...")
            self.run_command(["cargo", "test"], REPO_ROOT / "src-tauri")
        return {"detail": "Typecheck & test suite passed"}

    def run_build(self, version: str, build_type: str) -> dict:
        dist_dir = REPO_ROOT / "dist-release"
        if dist_dir.exists():
            for f in dist_dir.glob("*"):
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()

        timeout = 900 if build_type == "all" else 300
        self.run_command(["npm", "run", f"build:{build_type}"], timeout=timeout)
        artifacts = [a for a in (REPO_ROOT / "dist-release").glob("*") if a.is_file()]
        if not artifacts:
            raise Exception(
                "Build completed but no files were produced in dist-release"
            )
        names = ", ".join(a.name for a in artifacts)
        return {"detail": names}

    def run_git(self, version: str, build_type: str) -> dict:
        files = [
            "package.json",
            "package-lock.json",
            "src-tauri/Cargo.toml",
            "src-tauri/tauri.conf.json",
            "src-tauri/Cargo.lock",
        ]
        r = self.run_command(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"]
        )
        if r["success"] and not self.config.allow_dirty:
            dirty_other = []
            for line in r["output"].splitlines():
                if len(line) < 4:
                    continue
                path = line[3:].strip()
                if " -> " in path:
                    path = path.split(" -> ", 1)[-1].strip()
                if path not in files and not path.startswith("dist-release/"):
                    dirty_other.append(path)
            if dirty_other:
                raise DirtyRepoError(f"Uncommitted files: {', '.join(dirty_other)}")

        missing = [f for f in files if not (REPO_ROOT / f).exists()]
        if missing:
            raise Exception(f"Cannot stage missing files: {', '.join(missing)}")

        self.run_command(["git", "add", "--"] + files)
        try:
            self.run_command(["git", "commit", "-m", f"release: v{version}"])
        except Exception as e:
            raise Exception(
                f"Git commit failed; verify the version changed and Git identity is configured. {e}"
            ) from e

        r_tag = self.run_command(["git", "tag", "-l", f"v{version}"])
        if not r_tag["output"].strip():
            self.run_command(
                ["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"]
            )

        return {"detail": f"Tagged commit as v{version}"}

    def run_release(self, version: str, build_type: str) -> dict:
        self.run_command(["git", "push", "origin", self.config.git_branch])
        self.run_command(["git", "push", "origin", f"v{version}"])

        artifacts = list((REPO_ROOT / "dist-release").glob("*"))
        artifact_paths = [str(a) for a in artifacts]

        view_env = os.environ.copy()
        if LLVM_BIN:
            view_env["PATH"] = f"{LLVM_BIN}{os.pathsep}{view_env.get('PATH', '')}"
        view = subprocess.run(
            ["gh", "release", "view", f"v{version}", "--json", "name"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env=view_env,
        )
        if view.returncode != 0:
            self.run_command(
                ["gh", "release", "create", f"v{version}"]
                + artifact_paths
                + ["--title", f"v{version}", "--generate-notes"]
            )
        return {"detail": f"GitHub release v{version} published"}


# -------------------------------------------------------------------------
# Stylesheet
# -------------------------------------------------------------------------


class ReleaseApp(App):
    TITLE = "Release Orchestrator"
    CSS = """
    Screen {
        background: #0d1117;
        color: #c9d1d9;
    }

    Header {
        background: #161b22;
        color: #58a6ff;
        border-bottom: solid #30363d;
        height: 1;
    }

    Footer {
        background: #161b22;
        color: #8b949e;
    }

    .text-bold { text-style: bold; }
    .text-accent { color: #58a6ff; }
    .text-muted { color: #8b949e; }
    .mt-1 { margin-top: 1; }
    .hidden { display: none; }

.app-layout-centered {
        align: center middle;
        width: 100%;
        height: 100%;
        overflow-y: auto;
    }

    .config-dialog {
        width: 86;
        height: auto;
        max-height: 95%;
        overflow-y: auto;
        background: #161b22;
        border: round #30363d;
        padding: 1 2;
    }

    .dialog-header {
        height: 3;
        border-bottom: solid #30363d;
        align: center middle;
        margin-bottom: 1;
    }

    .branch-tag {
        background: #1f242c;
        color: #3fb950;
        border: solid #238636;
        padding: 0 1;
        text-style: bold;
    }

    .dialog-body {
        height: auto;
    }

    .dialog-col {
        width: 1fr;
        padding: 0 1;
    }

    .col-header {
        color: #f0f6fc;
        text-style: bold;
        border-bottom: solid #21262d;
        padding-bottom: 0;
        margin-bottom: 1;
    }

    .current-ver-card {
        background: #0d1117;
        padding: 0 1;
        height: 3;
        align: center middle;
        border: solid #21262d;
        margin-bottom: 1;
    }

    .input-caption {
        color: #6e7681;
        text-style: italic;
    }

    .quick-bump-row {
        height: 3;
        margin-bottom: 1;
    }

    .btn-bump {
        min-width: 11;
        height: 3;
        margin-right: 1;
        background: #21262d;
        color: #c9d1d9;
        border: none;
    }

    .btn-bump:hover {
        background: #30363d;
        color: #58a6ff;
    }

    Input {
        background: #0d1117;
        border: solid #30363d;
        color: #f0f6fc;
        margin-bottom: 1;
    }

    Input:focus {
        border: solid #58a6ff;
    }

    .affected-files-box {
        background: #0d1117;
        padding: 0 1;
        border: solid #21262d;
    }

    .mini-header {
        color: #8b949e;
        text-style: bold;
    }

    .file-entry {
        margin-left: 1;
    }

    RadioSet {
        background: transparent;
        border: none;
    }

    RadioButton {
        padding: 0;
        color: #c9d1d9;
    }

    Checkbox {
        background: transparent;
        border: none;
        color: #c9d1d9;
    }

    .dialog-actions {
        height: 3;
        margin-top: 1;
        border-top: solid #30363d;
        padding-top: 0;
        align: right middle;
    }

    .action-hint {
        width: 1fr;
        color: #484f58;
    }

    Button.-primary {
        background: #238636;
        color: #ffffff;
        text-style: bold;
        border: none;
    }

    Button.-primary:hover {
        background: #2ea043;
    }

    Button.-default {
        background: #21262d;
        color: #c9d1d9;
        border: none;
        margin-right: 1;
    }

    .cockpit-root {
        height: 100%;
        padding: 1 2;
    }

    .cockpit-topbar {
        height: 3;
        border-bottom: solid #30363d;
        align: center middle;
        margin-bottom: 1;
    }

    .version-badge {
        background: #1f6feb;
        color: #ffffff;
        text-style: bold;
        padding: 0 1;
        margin-left: 1;
    }

    .cockpit-body {
        height: 1fr;
    }

    .cockpit-left-pane {
        width: 44;
        height: 100%;
        background: #161b22;
        border: round #30363d;
        padding: 1;
        margin-right: 1;
    }

    .stepper-list {
        height: auto;
    }

    .step-row {
        height: 3;
        padding: 0 1;
        background: #0d1117;
        border: solid #21262d;
        margin-bottom: 1;
        align: center middle;
    }

    .step--active {
        border: solid #f0883e;
    }

    .step--done {
        border: solid #238636;
    }

    .step--error {
        border: solid #f85149;
    }

    .step-num {
        width: 3;
        color: #484f58;
        text-style: bold;
    }

    .step-text {
        width: 1fr;
    }

    .step-title {
        color: #f0f6fc;
        text-style: bold;
    }

    .step-desc {
        color: #8b949e;
    }

    .step-badge {
        text-style: bold;
        padding: 0 1;
    }

    .badge-pending { color: #8b949e; background: #21262d; }
    .badge-running { color: #ffffff; background: #d29922; }
    .badge-done    { color: #ffffff; background: #238636; }
    .badge-error   { color: #ffffff; background: #da3633; }

    ProgressBar {
        margin-top: 1;
        padding: 0;
    }

    #btn-force-continue {
        width: 100%;
        margin-top: 1;
        background: #d29922;
        color: #0d1117;
        text-style: bold;
        border: none;
    }

    .cockpit-right-pane {
        width: 1fr;
        height: 100%;
        background: #161b22;
        border: round #30363d;
        padding: 1;
    }

    .log-header-bar {
        height: 2;
        align: center middle;
    }

    .stream-tag {
        color: #3fb950;
        text-style: bold;
    }

    RichLog {
        background: #090d12;
        border: solid #21262d;
        color: #c9d1d9;
        height: 1fr;
        padding: 1;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(ConfigScreen())

    async def action_quit(self) -> None:
        self.exit()


def main():
    app = ReleaseApp()
    app.run()


if __name__ == "__main__":
    main()
