const mobileMenuButton =
    document.getElementById("mobileMenuButton");

const sidebar =
    document.getElementById("sidebar");

const sidebarOverlay =
    document.getElementById("sidebarOverlay");


function openSidebar() {

    if (!sidebar) {
        return;
    }

    sidebar.classList.add("open");

    if (sidebarOverlay) {
        sidebarOverlay.classList.add("active");
    }
}


function closeSidebar() {

    if (!sidebar) {
        return;
    }

    sidebar.classList.remove("open");

    if (sidebarOverlay) {
        sidebarOverlay.classList.remove("active");
    }
}


if (mobileMenuButton) {

    mobileMenuButton.addEventListener(
        "click",
        openSidebar
    );
}


if (sidebarOverlay) {

    sidebarOverlay.addEventListener(
        "click",
        closeSidebar
    );
}