---
name: electron-mcp-debug
description: Debug Electron app UI, renderer, console, network, and runtime issues using chrome-devtools MCP attached to an Electron remote debugging port and playwright MCP for deterministic reproduction.
compatibility: opencode
metadata:
  app_type: electron
  mcp: chrome-devtools
---
# Electron MCP Debugging Skill

The app needs to be run and fiddly UI problems debugged. 

Use this skill when the user asks to debug an Electron app, especially bugs involving:
- blank windows
- renderer crashes
- UI not updating
- failed network requests
- console errors
- hydration/runtime errors
- broken clicks, routing, dialogs, menus, auth, or IPC-driven UI behavior

When you're asked to debug the app directly:

- launch the app with `REMOTE_DEBUGGING_PORT=9222 npm run dev` to enable the Chrome DevTools Protocol (CDP) for debugging. (keep searching until you find an open port)
- Use the MCP to diagnose the issue
- Propose fixes, create tests, report back as requested by the user
