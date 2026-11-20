#!/usr/bin/env node

/**
 * Situs CLI - Distributed GPU Pooling Across the Internet
 * Node.js wrapper for cross-platform global installation via npm / npx.
 */

const { spawn, spawnSync } = require("child_process");
const process = require("process");
const path = require("path");
const fs = require("fs");

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 10;

function findPython() {
  const candidates = process.platform === "win32"
    ? ["python", "py", "python3"]
    : ["python3", "python"];

  for (const cmd of candidates) {
    try {
      const res = spawnSync(cmd, ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], {
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "ignore"],
      });
      if (res.status === 0 && res.stdout) {
        const [major, minor] = res.stdout.trim().split(".").map(Number);
        if (major > MIN_PYTHON_MAJOR || (major === MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR)) {
          return { command: cmd, version: `${major}.${minor}` };
        }
      }
    } catch {
      // Continue searching
    }
  }
  return null;
}

function isSitusInstalled(pythonCmd) {
  try {
    const res = spawnSync(pythonCmd, ["-c", "import situs; print(situs.__file__)"], {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    return res.status === 0;
  } catch {
    return false;
  }
}

function installSitus(pythonCmd) {
  console.log("\x1b[36m%s\x1b[0m", "📦 Situs Python engine not found in your environment.");
  
  // Check if running from local repo
  const repoRoot = path.resolve(__dirname, "..");
  const pyprojectPath = path.join(repoRoot, "pyproject.toml");
  const isLocalRepo = fs.existsSync(pyprojectPath);

  const installTarget = isLocalRepo ? repoRoot : "situs";
  console.log("\x1b[33m%s\x1b[0m", `⚙️  Installing Situs via pip (${installTarget})...`);

  const installArgs = ["-m", "pip", "install"];
  if (isLocalRepo) {
    installArgs.push("-e", repoRoot);
  } else {
    installArgs.push(installTarget);
  }

  let res = spawnSync(pythonCmd, installArgs, {
    stdio: "inherit",
  });

  // If pip install from PyPI fails (e.g. before PyPI release), fallback to GitHub test branch
  if (res.status !== 0 && !isLocalRepo) {
    console.log("\x1b[33m%s\x1b[0m", "🔄 Attempting install directly from GitHub repository (test branch)...");
    res = spawnSync(pythonCmd, ["-m", "pip", "install", "git+https://github.com/legalmonkey/situs.git@test"], {
      stdio: "inherit",
    });
  }

  if (res.status !== 0) {
    console.error("\x1b[31m%s\x1b[0m", "❌ Failed to install Situs dependencies via pip.");
    console.error(`Please install manually using: ${pythonCmd} -m pip install situs`);
    process.exit(res.status || 1);
  }

  console.log("\x1b[32m%s\x1b[0m", "✅ Situs engine installed successfully!\n");
}

function main() {
  const python = findPython();
  if (!python) {
    console.error("\x1b[31m%s\x1b[0m", "❌ Python 3.10 or higher is required to run Situs.");
    console.error("Please install Python from https://www.python.org/downloads/ and ensure it is in your PATH.");
    process.exit(1);
  }

  // Check if Situs is installed in the python environment
  if (!isSitusInstalled(python.command)) {
    installSitus(python.command);
  }

  // Forward all CLI arguments to Situs
  const args = ["-m", "situs", ...process.argv.slice(2)];

  const child = spawn(python.command, args, {
    stdio: "inherit",
    windowsHide: false,
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code || 0);
    }
  });

  process.on("SIGINT", () => {
    if (child.pid) {
      child.kill("SIGINT");
    }
  });

  process.on("SIGTERM", () => {
    if (child.pid) {
      child.kill("SIGTERM");
    }
  });
}

main();
