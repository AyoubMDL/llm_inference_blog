document$.subscribe(() => {
  document.querySelectorAll("a[href^='http']").forEach(link => {
    if (!link.hostname.includes(window.location.hostname)) {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
    }
  });
});
