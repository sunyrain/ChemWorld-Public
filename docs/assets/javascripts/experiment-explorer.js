(() => {
  "use strict";

  const familyFor = (action) => {
    const operation = action.action.operation;
    if (operation === "measure" && action.action.instrument === "final_assay") return "assay";
    if (operation === "measure") return "instrument";
    if (["heat", "wait"].includes(operation)) return "thermal";
    if (["quench", "evaporate", "distill", "collect_fraction"].includes(operation)) return "workup";
    if (operation === "terminate") return "terminal";
    return "setup";
  };

  const number = (value, digits = 3) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(digits);
  };

  const zhLabels = {
    Reagent: "试剂",
    Solvent: "溶剂",
    Catalyst: "催化剂",
    Heat: "加热",
    Wait: "等待",
    Quench: "淬灭",
    Evaporate: "蒸发",
    Distill: "蒸馏",
    Fraction: "收集馏分",
    Terminate: "终止",
    "Final assay": "终检",
  };

  const localizedAction = (action, locale) => locale === "zh" ? (zhLabels[action.label] || action.label) : action.label;

  const localizedDetails = (value, locale) => {
    if (locale !== "zh") return value;
    return value
      .replace(/solvent /g, "溶剂 ")
      .replace(/catalyst /g, "催化剂 ")
      .replace(/fraction /g, "比例 ")
      .replace(/reflux /g, "回流比 ");
  };

  const localizedObservation = (value, locale) => {
    if (locale !== "zh") return value;
    return value
      .replace(/distillate recovery/g, "馏出物回收率")
      .replace(/distillate purity/g, "馏出物纯度")
      .replace(/endpoint score/g, "终点得分")
      .replace(/conversion/g, "转化率")
      .replace(/selectivity/g, "选择性")
      .replace(/yield/g, "产率")
      .replace(/score/g, "得分");
  };

  const text = (tag, value, className) => {
    const node = document.createElement(tag);
    node.textContent = value;
    if (className) node.className = className;
    return node;
  };

  const detailList = (pairs) => {
    const list = document.createElement("dl");
    list.className = "cw-key-value";
    pairs.forEach(([key, value]) => {
      list.append(text("dt", key), text("dd", String(value)));
    });
    return list;
  };

  const render = (root, payload) => {
    const locale = root.dataset.locale || "en";
    const lifecycle = payload.agent_lifecycle;
    const actions = lifecycle.actions;
    const strip = root.querySelector("[data-cw-step-strip]");
    const detail = root.querySelector("[data-cw-step-detail]");
    const summary = root.querySelector("[data-cw-summary]");

    summary.textContent = locale === "zh"
      ? `${lifecycle.summary.committed_actions} 个提交动作 · ${number(lifecycle.summary.process_time_used_s, 0)} / ${number(lifecycle.summary.process_time_limit_s, 0)} 过程秒 · 精确回放`
      : `${lifecycle.summary.committed_actions} committed actions · ${number(lifecycle.summary.process_time_used_s, 0)} / ${number(lifecycle.summary.process_time_limit_s, 0)} process seconds · exact replay`;

    const select = (index) => {
      const action = actions[index];
      strip.querySelectorAll("button").forEach((button, buttonIndex) => {
        button.setAttribute("aria-pressed", buttonIndex === index ? "true" : "false");
      });
      detail.replaceChildren();

      const actionPanel = document.createElement("section");
      actionPanel.className = "cw-step-panel";
      actionPanel.append(text("h4", `${locale === "zh" ? "步骤" : "Step"} ${action.step} · ${localizedAction(action, locale)}`));
      actionPanel.append(detailList([
        [locale === "zh" ? "公开参数" : "Public parameters", localizedDetails(action.details, locale)],
        [locale === "zh" ? "事务" : "Transaction", locale === "zh" && action.transaction_status === "committed" ? "已提交" : action.transaction_status],
        [locale === "zh" ? "过程时间增量" : "Process-time delta", `${number(action.process_time_delta_s)} s`],
        [locale === "zh" ? "累计时间" : "Cumulative time", `${number(action.cumulative_process_time_s)} s`],
      ]));

      const observationPanel = document.createElement("section");
      observationPanel.className = "cw-step-panel";
      observationPanel.append(text("h4", locale === "zh" ? "公开效果" : "Public effect"));
      observationPanel.append(text("p", localizedObservation(action.observation_highlight, locale)));
      const track = document.createElement("div");
      track.className = "cw-score-track";
      track.setAttribute("aria-label", "Public score relative to the final displayed score");
      const fill = document.createElement("div");
      fill.className = "cw-score-fill";
      const score = action.public_score === null ? 0 : Number(action.public_score);
      const maxScore = Number(lifecycle.final_public_observation.score) || 1;
      fill.style.width = `${Math.max(1, Math.min(100, (score / maxScore) * 100))}%`;
      track.append(fill);
      observationPanel.append(track, text("small", `${locale === "zh" ? "公开得分" : "Public score"}: ${number(action.public_score)}`));

      detail.append(actionPanel, observationPanel);
    };

    actions.forEach((action, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "cw-step-button";
      button.dataset.family = familyFor(action);
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `${locale === "zh" ? "显示步骤" : "Show step"} ${action.step}: ${localizedAction(action, locale)}`);
      button.append(text("span", String(action.step), "cw-step-number"));
      button.append(text("span", localizedAction(action, locale), "cw-step-label"));
      button.addEventListener("click", () => select(index));
      button.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const target = event.key === "Home" ? 0 : event.key === "End" ? actions.length - 1 :
          (index + (event.key === "ArrowRight" ? 1 : -1) + actions.length) % actions.length;
        strip.children[target].focus();
        select(target);
      });
      strip.append(button);
    });
    select(0);
  };

  const initialize = async (root) => {
    try {
      const response = await fetch(root.dataset.source, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(root, await response.json());
    } catch (error) {
      const target = root.querySelector("[data-cw-step-detail]");
      target.replaceChildren(text("p", `The interactive view could not load (${error.message}). The complete static table remains below.`, "cw-error"));
    }
  };

  const boot = () => document.querySelectorAll("[data-cw-explorer]").forEach(initialize);
  const initializeAutoplay = async (root) => {
    try {
      const response = await fetch(root.dataset.source, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      const actions = payload.agent_lifecycle.actions;
      const locale = root.dataset.locale || "en";
      const rail = root.querySelector("[data-cw-live-rail]");
      const label = root.querySelector("[data-cw-live-label]");
      const detail = root.querySelector("[data-cw-live-detail]");
      const family = root.querySelector("[data-cw-live-family]");
      const score = root.querySelector("[data-cw-live-score]");
      const step = root.querySelector("[data-cw-live-step]");
      const progress = root.querySelector("[data-cw-live-progress]");
      const familyNames = locale === "zh"
        ? { setup: "准备", thermal: "过程", instrument: "仪器", workup: "后处理", terminal: "终止", assay: "终检" }
        : { setup: "SETUP", thermal: "PROCESS", instrument: "INSTRUMENT", workup: "WORKUP", terminal: "TERMINAL", assay: "ASSAY" };
      let active = 0;
      let timer = null;

      const select = (index) => {
        active = index;
        const action = actions[index];
        const actionFamily = familyFor(action);
        rail.querySelectorAll("button").forEach((button, buttonIndex) => {
          button.classList.toggle("is-active", buttonIndex === index);
          button.setAttribute("aria-pressed", buttonIndex === index ? "true" : "false");
        });
        label.textContent = localizedAction(action, locale);
        detail.textContent = localizedDetails(action.details, locale);
        family.textContent = familyNames[actionFamily];
        score.textContent = number(action.public_score);
        step.textContent = String(action.step).padStart(2, "0");
        progress.style.width = `${(action.step / actions.length) * 100}%`;
        root.dataset.activeFamily = actionFamily;
      };

      const play = () => {
        window.clearInterval(timer);
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
        timer = window.setInterval(() => select((active + 1) % actions.length), 2200);
      };

      actions.forEach((action, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.family = familyFor(action);
        button.setAttribute("aria-pressed", "false");
        button.setAttribute("aria-label", `${locale === "zh" ? "显示步骤" : "Show step"} ${action.step}: ${localizedAction(action, locale)}`);
        button.innerHTML = `<span>${String(action.step).padStart(2, "0")}</span><i></i>`;
        button.addEventListener("click", () => select(index));
        rail.append(button);
      });
      select(0);
      play();
      root.addEventListener("mouseenter", () => window.clearInterval(timer));
      root.addEventListener("mouseleave", play);
      root.addEventListener("focusin", () => window.clearInterval(timer));
      root.addEventListener("focusout", play);
    } catch (error) {
      root.classList.add("cw-autoplay-error");
    }
  };

  const bootAll = () => {
    boot();
    document.querySelectorAll("[data-cw-autoplay]").forEach(initializeAutoplay);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bootAll);
  else bootAll();
})();
