"use strict";

(() => {
  let session = null;

  function disconnect() {
    if (session) session.close();
    session = null;
  }

  function connect() {
    if (session) return;
    session = new EventSource("/api/ui/session");
    session.addEventListener("shutdown", disconnect);
  }

  window.addEventListener("pagehide", disconnect);
  window.addEventListener("pageshow", connect);
  connect();
})();
