(() => {
"use strict";

const byId = (id) => document.getElementById(id);
const token = () => localStorage.getItem("access_token");

async function apiRequest(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const accessToken = token();
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeout || 12000);
    try {
        const response = await fetch(path, {
            ...options,
            headers,
            signal: options.signal || controller.signal,
        });
        const contentType = response.headers.get("content-type") || "";
        const payload = contentType.includes("application/json")
            ? await response.json()
            : await response.text();
        if (!response.ok) {
            const detail = payload && typeof payload === "object" ? payload.detail : null;
            const error = new Error(detail || `Request failed (${response.status}).`);
            error.status = response.status;
            throw error;
        }
        return payload;
    } catch (error) {
        if (error.name === "AbortError") throw new Error("The request timed out. Check the connection and retry.");
        if (error instanceof TypeError) throw new Error("Could not reach SmartAttend. Check that the server is running.");
        throw error;
    } finally {
        window.clearTimeout(timeout);
    }
}

window.SmartAttend = { api: apiRequest, token };
try { window.SmartAttend.currentUser = JSON.parse(localStorage.getItem("user") || "null"); } catch (_) { window.SmartAttend.currentUser = null; }

const menuButton = byId("mobileMenuButton");
const sidebar = byId("sidebar");
const overlay = byId("sidebarOverlay");
function closeSidebar() {
    sidebar?.classList.remove("open");
    overlay?.classList.remove("active");
    menuButton?.setAttribute("aria-expanded", "false");
    menuButton?.setAttribute("aria-label", "Open navigation");
}
menuButton?.addEventListener("click", () => {
    const opening = !sidebar?.classList.contains("open");
    sidebar?.classList.toggle("open", opening);
    overlay?.classList.toggle("active", opening);
    menuButton.setAttribute("aria-expanded", String(opening));
    menuButton.setAttribute("aria-label", opening ? "Close navigation" : "Open navigation");
});
overlay?.addEventListener("click", closeSidebar);
sidebar?.querySelectorAll("a.nav-item").forEach((link) => link.addEventListener("click", closeSidebar));
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSidebar();
});

function clearSession() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
}
document.querySelectorAll("[data-logout]").forEach((button) => {
    button.addEventListener("click", () => {
        clearSession();
        window.location.assign("/login");
    });
});

function dashboardForRole(role) {
    return ({ admin: "/admin-dashboard", teacher: "/teacher-dashboard", student: "/student-dashboard" })[role] || "/login";
}

async function loadAuthenticatedUser() {
    const requiredRoles = (document.body.dataset.requiredRoles || "").split(",").filter(Boolean);
    const identity = document.querySelector("[data-auth-user]");
    const roleNode = document.querySelector("[data-auth-role]");
    const status = byId("authStatus");
    const accessToken = token();
    if (!accessToken) {
        if (identity) identity.textContent = "Not signed in";
        if (roleNode) roleNode.textContent = "";
        if (status) status.textContent = "Your session ended. Please sign in again.";
        if (requiredRoles.length) window.location.replace("/login");
        return;
    }
    try {
        const user = await apiRequest("/api/me");
        window.SmartAttend.currentUser = user;
        if (identity) identity.textContent = user.email;
        if (roleNode) roleNode.textContent = user.role;
        if (status) status.textContent = "";
        if (user.must_change_password && window.location.pathname !== "/profile") {
            window.location.replace("/profile?first_login=1");
            return;
        }
        document.querySelectorAll("[data-app-home]").forEach((link) => {
            link.href = dashboardForRole(user.role);
        });
        document.querySelectorAll("[data-auth-avatar]").forEach((node) => {
            node.textContent = user.email.slice(0, 1).toUpperCase();
        });
        localStorage.setItem("user", JSON.stringify(user));

        const allowedNavigation = {
            "/admin-dashboard": ["admin"],
            "/teacher-dashboard": ["admin", "teacher"],
            "/student-dashboard": ["student"],
            "/students": ["admin", "teacher"],
            "/take-attendance": ["admin", "teacher"],
            "/attendance-records": ["admin", "teacher", "student"],
            "/analytics": ["admin", "teacher", "student"],
        };
        document.querySelectorAll("a[href]").forEach((link) => {
            const path = new URL(link.href, window.location.origin).pathname;
            if (allowedNavigation[path]) link.hidden = !allowedNavigation[path].includes(user.role);
        });
        document.querySelectorAll(".sidebar-nav .nav-section").forEach((section) => {
            section.hidden = ![...section.querySelectorAll("a[href]")].some((link) => !link.hidden);
        });
        if (requiredRoles.length && !requiredRoles.includes(user.role)) {
            window.location.replace(dashboardForRole(user.role));
        }
        if (user.must_change_password) {
            document.querySelectorAll("#currentPassword").forEach((input) => { input.required = true; });
            const notice = document.querySelector("[data-first-login]");
            if (notice) notice.hidden = false;
        }
    } catch (error) {
        if (identity) identity.textContent = error.status === 401 ? "Sign in required" : "Account unavailable";
        if (roleNode) roleNode.textContent = "";
        if (status) {
            status.replaceChildren(document.createTextNode(error.message));
            if (error.status !== 401) {
                const retry = document.createElement("button");
                retry.type = "button";
                retry.className = "button button-secondary";
                retry.textContent = "Retry account check";
                retry.addEventListener("click", loadAuthenticatedUser, { once: true });
                status.append(" ", retry);
            }
        }
        if (error.status === 401) {
            clearSession();
            if (requiredRoles.length) window.location.replace("/login");
        }
    }
}

async function loadTeacherDashboard() {
    const stats = document.querySelectorAll(".stats-grid [data-stat]");
    const sessionList = byId("dashboardSessions");
    const chart = byId("dashboardTrend");
    const defaulters = byId("dashboardDefaulters");
    const status = byId("dashboardStatus");
    const refreshButton = byId("refreshTeacherDashboard");
    if (!stats.length && !sessionList && !chart && !defaulters) return;

    stats.forEach((node) => { node.textContent = "—"; });
    if (sessionList) sessionList.textContent = "Loading recent sessions…";
    if (chart) chart.textContent = "Loading saved check-ins…";
    if (defaulters) defaulters.textContent = "Loading attendance coverage…";
    if (status) status.textContent = "";

    const render = async () => {
        if (refreshButton) {
            refreshButton.disabled = true;
            refreshButton.textContent = "Refreshing…";
        }
        if (status) status.replaceChildren();
        try {
            const data = await apiRequest("/api/dashboard");
            stats.forEach((node) => {
                const value = data[node.dataset.stat];
                node.textContent = value === null || value === undefined ? "—" : value;
            });
            if (sessionList) {
                sessionList.replaceChildren();
                if (!data.recent_sessions.length) sessionList.textContent = "No sessions yet. Start one when your class is ready.";
                data.recent_sessions.forEach((session) => {
                    const item = document.createElement("div");
                    item.className = "session-item";
                    const title = document.createElement("strong");
                    title.textContent = session.subject;
                    const detail = document.createElement("span");
                    detail.textContent = `${session.date} · ${session.check_ins} present · ${session.status}`;
                    item.append(title, detail);
                    sessionList.appendChild(item);
                });
            }
            if (chart) {
                chart.replaceChildren();
                const days = data.daily_check_ins || [];
                if (!days.some((item) => item.count)) {
                    chart.textContent = "No saved check-ins in the last 14 days.";
                } else {
                    const max = Math.max(1, ...days.map((item) => item.count));
                    const bars = document.createElement("div");
                    bars.className = "attendance-bars";
                    days.forEach((day) => {
                        const column = document.createElement("div");
                        column.className = "attendance-bar-column";
                        const value = document.createElement("span");
                        value.textContent = day.count;
                        const bar = document.createElement("div");
                        bar.className = "attendance-bar";
                        bar.style.height = `${Math.max(day.count ? 8 : 2, day.count / max * 120)}px`;
                        bar.title = `${day.date}: ${day.count} saved check-ins`;
                        const label = document.createElement("small");
                        label.textContent = day.date.slice(5);
                        column.append(value, bar, label);
                        bars.appendChild(column);
                    });
                    chart.appendChild(bars);
                }
            }
            if (defaulters) {
                defaulters.replaceChildren();
                const candidates = data.defaulters || [];
                if (!data.attendance_denominator) {
                    defaulters.textContent = "Attendance risk appears after a session is closed. No completed sessions yet.";
                } else if (!candidates.length) {
                    defaulters.textContent = `No students below ${data.attendance_threshold_percent}% across completed sessions.`;
                } else {
                    candidates.slice(0, 8).forEach((student) => {
                        const item = document.createElement("li");
                        item.textContent = `${student.student_name} (${student.roll_number}) · ${student.attendance_percentage}%`;
                        defaulters.appendChild(item);
                    });
                }
            }
        } catch (error) {
            stats.forEach((node) => { node.textContent = "—"; });
            if (sessionList) sessionList.textContent = error.message;
            if (chart) chart.textContent = error.message;
            if (defaulters) defaulters.textContent = error.message;
            if (status) {
                status.textContent = error.message;
                const retry = document.createElement("button");
                retry.type = "button";
                retry.className = "secondary-button";
                retry.textContent = "Retry";
                retry.addEventListener("click", render, { once: true });
                status.append(" ", retry);
            }
        } finally {
            if (refreshButton) {
                refreshButton.disabled = false;
                refreshButton.textContent = "Refresh dashboard";
            }
        }
    };
    refreshButton?.addEventListener("click", render);
    await render();
}

loadAuthenticatedUser();
loadTeacherDashboard();
})();
