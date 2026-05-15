document.addEventListener('DOMContentLoaded', () => {
    // Theme Management
    const themeBtn = document.getElementById('theme-toggle');
    const root = document.documentElement;

    function applyTheme(theme) {
        if (theme === 'dark') { root.setAttribute('data-theme', 'dark'); if(themeBtn) themeBtn.textContent = '☀️'; } 
        else { root.removeAttribute('data-theme'); if(themeBtn) themeBtn.textContent = '🌙'; }
        localStorage.setItem('theme', theme);
        
        // Notify V-Graph iframes
        document.querySelectorAll('iframe').forEach(ifr => {
            if(ifr.contentWindow) ifr.contentWindow.postMessage('theme-toggled', '*');
        });
    }

    const savedTheme = localStorage.getItem('theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = root.getAttribute('data-theme');
            applyTheme(current === 'dark' ? 'light' : 'dark');
        });
    }

    // Submit Validation
    const proceedBtn = document.getElementById('proceed-btn');
    if (proceedBtn) {
        proceedBtn.addEventListener('click', function(e) {
            const datFile = document.getElementById('file-dat').files.length;
            if (!datFile) { alert("Please upload a Measurement File (.DAT) to proceed."); return; }
            
            document.getElementById('upload-container-main').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('upload-container-main').style.display = 'none';
                document.getElementById('loader').style.display = 'block';
            }, 200);
            document.getElementById('upload-form').submit();
        });
    }
});

// Drag and Drop Logic
function handleDrop(e, inputId, pillId) {
    e.preventDefault();
    const dt = e.dataTransfer;
    if (dt.files.length) {
        document.getElementById(inputId).files = dt.files;
        updatePill(inputId, pillId);
    }
}
function updatePill(inputId, pillId) {
    const inp = document.getElementById(inputId);
    const pill = document.getElementById(pillId);
    if (inp.files.length > 0) {
        pill.style.display = 'inline-flex';
        pill.innerHTML = `<span class="dot"></span> ${inp.files[0].name}`;
    }
}

// Tab Layout Sync Logic
function openTab(evt, tabName) {
    var i, x, tablinks;
    x = document.getElementsByClassName("tab-content");
    for (i = 0; i < x.length; i++) { x[i].style.display = "none"; }
    tablinks = document.getElementsByClassName("tab-btn");
    for (i = 0; i < x.length; i++) { tablinks[i].classList.remove("active"); }
    document.getElementById(tabName).style.display = "flex";
    evt.currentTarget.classList.add("active");

    // Force iframe canvas to re-calculate its width when un-hidden
    const activeIframe = document.querySelector(`#${tabName} iframe`);
    if(activeIframe && activeIframe.contentWindow) {
        activeIframe.contentWindow.postMessage('tab-activated', '*');
    }
}