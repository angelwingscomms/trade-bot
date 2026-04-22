// Fresh Plugin
// Documentation: https://github.com/user/fresh/blob/main/docs/plugins.md

const editor = getEditor();

// Define a command handler and register it
function hello(): void {
  editor.setStatus("Hello from your plugin!");
}
registerHandler("hello", hello);
editor.registerCommand("hello", "Say Hello", "hello");

// React to editor events
function onBufferOpened(): void {
  const bufferId = editor.getActiveBufferId();
  const info = editor.getBufferInfo(bufferId);
  if (info) {
    editor.debug(`Opened: ${info.path}`);
  }
}
registerHandler("on_buffer_opened", onBufferOpened);
editor.on("buffer_opened", "on_buffer_opened");

// Example: Add a keybinding in your Fresh config:
// {
//   "keyBindings": {
//     "ctrl+alt+h": "command:hello"
//   }
// }
