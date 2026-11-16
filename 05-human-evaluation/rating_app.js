/* Independent local ratings. No network, default scores, or source model labels. */
(function () {
  'use strict';
  const el = id => document.getElementById(id);
  try {
    const packet = window.RATING_PACKET, delivery = window.DELIVERY_STATE, config = window.RATING_CONFIG;
    if (!packet || !delivery || !config || !window.fflate || !window.RatingWorkbook) throw Error('评分程序文件不完整，请完整解压评分包。');
    if (packet.batch_id !== config.batch_id || delivery.batch_id !== config.batch_id || packet.rater !== config.rater) throw Error('评分包批次或评审身份不匹配。');
    if (config.rating_policy !== 'scores-only-v1') throw Error('评分界面版本不完整，请使用最新完整包。');
    const rows = packet.rows, keys = config.scores, rowById = new Map(rows.map(r => [r.clip_id, r]));
    if (rows.length !== 96 || rowById.size !== 96 || rows.some(r => !delivery.clips[r.clip_id])) throw Error('评分包应保留96个有效固定位置。');
    const TIMES = {start: ['前段', 0, 1 / 3], middle: ['中段', 1 / 3, 2 / 3], end: ['后段', 2 / 3, 1], all: ['全程', 0, 1]};
    const storageKey = 'ai6132-human:' + config.batch_id + ':' + packet.rater;
    const session = Math.random().toString(36).slice(2);
    const blankRecord = () => ({scores: Object.fromEntries(keys.map(k => [k, ''])), issues: {}, time_ranges: [],
      custom_start: '', custom_end: '', na_reason: '', manual_time: '', notes: '', confirmed_failure: false});
    const newDraft = () => ({version: config.version, protocol: config.protocol, batch_id: config.batch_id,
      rater: packet.rater, records: {}, declarations: {}, position: 0, revision: null});
    let draft = newDraft(), position = 0, savedRevision = null, storageWorks = true, conflict = false;
    const recordFor = clip => draft.records[clip] || blankRecord();
    const stateFor = clip => delivery.clips[clip];
    function touched(record) {
      return Object.values(record.scores).some(Boolean) || Object.values(record.issues).some(v => v.length) || record.time_ranges.length ||
        record.custom_start !== '' || record.custom_end !== '' || record.na_reason || record.manual_time || record.notes || record.confirmed_failure;
    }
    function duration(clip) {
      const known = stateFor(clip).duration_seconds || config.durations[clip];
      if (Number.isFinite(known) && known > 0) return known;
      return rows[position].clip_id === clip && Number.isFinite(el('video').duration) ? el('video').duration : null;
    }
    function timeText(record, clip) {
      const total = duration(clip), pieces = [];
      if (total && record.time_ranges.length) {
        const ranges = record.time_ranges.includes('all') ? ['all'] : record.time_ranges;
        for (const key of ranges) pieces.push((TIMES[key][1] * total).toFixed(2) + '–' + (TIMES[key][2] * total).toFixed(2) + '秒');
      }
      if (record.custom_start !== '' && record.custom_end !== '') pieces.push(record.custom_start + '–' + record.custom_end + '秒');
      if (record.manual_time.trim()) pieces.push(record.manual_time.trim());
      return pieces.join('；');
    }
    function commentText(record, clip) {
      const comments = config.rubrics.flatMap(rubric => {
        const selected = record.issues[rubric.key] || [];
        return selected.length ? [rubric.label + '：' + selected.join('、')] : [];
      });
      if (record.confirmed_failure && stateFor(clip).status !== 'success' && stateFor(clip).status !== 'pending_delivery')
        comments.push('已核对技术失败：' + stateFor(clip).failure_reason);
      if (record.notes.trim()) comments.push(record.notes.trim());
      return comments.join('；');
    }
    function validateRecord(record, clip, required = false) {
      const status = stateFor(clip).status, errors = [];
      if (status === 'pending_delivery') return touched(record) ? ['尚未交付，所有分数与观察必须留空。'] : required ? ['素材尚未交付。'] : [];
      if (status !== 'success') {
        if (Object.values(record.scores).some(Boolean)) errors.push('技术失败的四项分数必须留空。');
        return errors;
      }
      if (!Object.values(record.scores).some(Boolean) && !required) return [];
      keys.forEach((key, index) => {
        const valid = ['1', '2', '3', '4', '5', ...(index === 3 ? ['NA'] : [])];
        if (!valid.includes(record.scores[key])) errors.push(config.rubrics[index].label + '尚未评分。');
      });
      const background = record.scores[keys[3]];
      if (record.na_reason && !config.na_reasons.includes(record.na_reason)) errors.push('背景NA原因无效。');
      if (background !== 'NA' && record.na_reason) errors.push('背景未填NA，请清除NA原因。');
      return errors;
    }
    function completed(clip) { return stateFor(clip).status !== 'pending_delivery' && !validateRecord(recordFor(clip), clip, true).length; }
    function normalizeImport(data) {
      if (!data || ![config.version, 'human-browser-v1'].includes(data.version) || data.protocol !== config.protocol || data.batch_id !== config.batch_id || data.rater !== packet.rater)
        throw Error('进度文件的版本、批次或评审身份不匹配。');
      if (!data.records || typeof data.records !== 'object' || Array.isArray(data.records)) throw Error('进度记录格式不正确。');
      const clean = newDraft();
      for (const [clip, source] of Object.entries(data.records)) {
        if (!rowById.has(clip) || !source || typeof source !== 'object' || Array.isArray(source)) throw Error('存在未知匿名编号或无效记录。');
        const record = blankRecord();
        for (const key of Object.keys(source.scores || {})) if (!keys.includes(key)) throw Error('存在未知评分维度。');
        keys.forEach((key, index) => {
          const value = String(source.scores?.[key] ?? '');
          if (!['', '1', '2', '3', '4', '5', ...(index === 3 ? ['NA'] : [])].includes(value)) throw Error(clip + '存在无效分数。');
          record.scores[key] = value;
          const issues = source.issues?.[key] || [];
          if (!Array.isArray(issues) || issues.some(x => !config.issue_options[key].includes(x)) || new Set(issues).size !== issues.length) throw Error('问题选项无效。');
          record.issues[key] = issues.slice();
        });
        for (const key of ['custom_start', 'custom_end', 'na_reason', 'manual_time', 'notes']) {
          const value = source[key] ?? '';
          if (typeof value !== 'string' || value.length > 10000) throw Error('进度文字字段无效或过长。');
          record[key] = value;
        }
        if (record.na_reason && !config.na_reasons.includes(record.na_reason)) throw Error('背景NA原因无效。');
        if (!Array.isArray(source.time_ranges) || source.time_ranges.some(key => !Object.hasOwn(TIMES, key))) throw Error('问题时段选项无效。');
        record.time_ranges = [...new Set(source.time_ranges)];
        record.confirmed_failure = source.confirmed_failure === true;
        if (stateFor(clip).status === 'pending_delivery' && touched(record)) throw Error('不能为未交付素材导入评分。');
        clean.records[clip] = record;
      }
      for (const key of ['name', 'date', 'seen_model_scores', 'discussed_video_defects', 'other']) {
        const value = data.declarations?.[key] ?? '';
        if (typeof value !== 'string' || value.length > 10000) throw Error('评审声明格式无效。');
        if (['seen_model_scores', 'discussed_video_defects'].includes(key) && !['', '是', '否'].includes(value)) throw Error('评审声明须为是/否。');
        clean.declarations[key] = value;
      }
      clean.position = Number.isInteger(data.position) && data.position >= 0 && data.position < rows.length ? data.position : 0;
      clean.revision = typeof data.revision === 'string' ? data.revision : null;
      return clean;
    }
    function message(text, kind = 'notice', errors = []) {
      const box = el('messages'); box.replaceChildren();
      if (!text) return;
      const p = document.createElement('div'); p.className = kind; p.textContent = text; box.appendChild(p);
      if (errors.length) {
        const list = document.createElement('ul'); list.className = 'error-list';
        errors.slice(0, 25).forEach(item => {
          const li = document.createElement('li'), a = document.createElement('a');
          a.href = '#' + item.clip; a.textContent = item.clip + '：' + item.text; li.appendChild(a); list.appendChild(li);
        }); box.appendChild(list);
      }
    }
    function save() {
      draft.position = position;
      if (conflict) { el('saveStatus').textContent = '检测到另一窗口更新，请先下载本页JSON备份，再重新打开页面或导入所需版本。'; return; }
      try {
        const latest = localStorage.getItem(storageKey);
        if (latest && JSON.parse(latest).revision !== savedRevision) { conflict = true; save(); return; }
        const candidate = {...draft, revision: Date.now() + ':' + session + ':' + Math.random().toString(36).slice(2)};
        localStorage.setItem(storageKey, JSON.stringify(candidate)); draft.revision = candidate.revision; savedRevision = candidate.revision;
        storageWorks = true;
        el('saveStatus').textContent = '本机浏览器草稿已保存 · ' + new Date().toLocaleTimeString() + '；结束时请下载JSON备份。';
      } catch (_) {
        storageWorks = false;
        el('saveStatus').textContent = '浏览器未能持久保存；当前内容仍在页面中，请立即下载JSON备份或导出Excel。';
      }
    }
    function download(bytes, name, mime) {
      const url = URL.createObjectURL(new Blob([bytes], {type: mime})), a = document.createElement('a');
      a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    const stamp = () => new Date().toISOString().replace(/[:.]/g, '-');
    function refresh() {
      let count = 0, available = 0, pending = 0;
      rows.forEach((row, index) => {
        const status = stateFor(row.clip_id).status, done = completed(row.clip_id), r = recordFor(row.clip_id);
        if (status === 'pending_delivery') pending++; else available++;
        if (done) count++;
        const label = status === 'pending_delivery' ? '待补充' : done ? '已完成' : touched(r) ? '待完善' : status === 'success' ? '未评分' : '技术失败';
        el('choose').options[index].textContent = String(index + 1).padStart(2, '0') + ' · ' + row.clip_id + ' · ' + label;
      });
      el('progress').textContent = '已完成 ' + count + ' / ' + available + ' · 待补充 ' + pending;
      el('exportFinal').disabled = pending > 0;
      el('exportFinal').title = pending ? '仍有素材待交付，可先导出当前评分。' : '全部视频完成四项评分后导出。';
      const clip = rows[position].clip_id, record = recordFor(clip), errors = validateRecord(record, clip, true);
      el('recordStatus').textContent = completed(clip) ? '本条已完成，已纳入进度。' : errors.join(' ');
    }
    function readCurrent() {
      const clip = rows[position].clip_id, status = stateFor(clip).status;
      if (status === 'pending_delivery') return;
      // Keep imported legacy notes/times privately; only scores/options are edited here.
      const record = {...recordFor(clip), scores: {}, issues: {}};
      keys.forEach(key => {
        record.scores[key] = document.querySelector('input[data-score="' + key + '"]:checked')?.value || '';
        record.issues[key] = Array.from(document.querySelectorAll('input[data-issue="' + key + '"]:checked')).map(input => input.value);
      });
      record.na_reason = el('naReason').value;
      const na = record.scores[keys[3]] === 'NA';
      el('naBox').classList.toggle('hide', !na);
      if (!na) { record.na_reason = ''; el('naReason').value = ''; }
      if (na) {
        record.issues[keys[3]] = [];
        document.querySelectorAll('input[data-issue="' + keys[3] + '"]').forEach(input => { input.checked = false; input.disabled = true; });
      } else document.querySelectorAll('input[data-issue="' + keys[3] + '"]').forEach(input => input.disabled = status !== 'success');
      draft.records[clip] = record; save(); refresh();
    }
    function show(index) {
      position = index;
      const row = rows[index], clip = row.clip_id, state = stateFor(clip), record = recordFor(clip), v = el('video');
      el('choose').value = String(index); el('clip').textContent = '观看序号 ' + (index + 1) + ' / 96 · ' + clip;
      el('prompt').textContent = row.prompt; el('reference').src = row.image_path; el('imageLink').href = row.image_path;
      v.pause(); v.removeAttribute('src'); v.load(); el('videoError').replaceChildren(); el('speed').value = '1';
      el('videoLink').href = row.video_path; el('videoLink').style.display = state.status === 'success' ? 'inline' : 'none';
      el('status').className = 'notice';
      el('status').textContent = state.status === 'success' ? '素材可用。先完整观看，再独立选择四项分数。' : state.status === 'pending_delivery' ?
        '等待后续交付：本位置留空，仍使用原编号和原进度。' : '技术失败：' + state.failure_reason + '。四项分数留空。';
      if (state.status === 'success') { v.src = row.video_path; v.playbackRate = 1; }
      el('ratingFields').disabled = state.status !== 'success';
      keys.forEach(key => {
        document.querySelectorAll('input[data-score="' + key + '"]').forEach(input => input.checked = input.value === record.scores[key]);
        document.querySelectorAll('input[data-issue="' + key + '"]').forEach(input => { input.checked = (record.issues[key] || []).includes(input.value); input.disabled = key === keys[3] && record.scores[key] === 'NA'; });
      });
      el('naReason').value = record.na_reason; el('naBox').classList.toggle('hide', record.scores[keys[3]] !== 'NA');
      location.hash = clip; refresh();
    }
    function move(direction) {
      let index = position;
      for (let n = 0; n < rows.length; n++) {
        index = (index + direction + rows.length) % rows.length;
        if (el('readyOnly').checked && stateFor(rows[index].clip_id).status === 'pending_delivery') continue;
        if (el('unfinishedOnly').checked && completed(rows[index].clip_id)) continue;
        show(index); return;
      }
      message('没有符合当前导航条件的其他条目。');
    }
    function exportExcel(final) {
      const errors = [];
      rows.forEach(row => validateRecord(recordFor(row.clip_id), row.clip_id, final).forEach(text => errors.push({clip: row.clip_id, text})));
      if (errors.length) { message('尚有 ' + errors.length + ' 项需要完善；也可先下载JSON备份保留未完成输入。', 'error', errors); return; }
      const values = rows.map(row => {
        const record = recordFor(row.clip_id);
        return {...record.scores, background_na_reason: record.na_reason, evidence_time: timeText(record, row.clip_id), comment: commentText(record, row.clip_id)};
      });
      const bytes = window.RatingWorkbook.exportWorkbook(config, values, draft.declarations);
      download(bytes, 'ratings_' + packet.rater + '_' + (final ? 'final' : 'current') + '_' + stamp() + '.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
      message('Excel已导出，含96行四项评分。请保留首次提交和JSON备份。', 'success');
    }
    config.rubrics.forEach(rubric => {
      const card = document.createElement('div'); card.className = 'score-card'; card.dataset.metric = rubric.key;
      const heading = document.createElement('h3'); heading.textContent = rubric.label; card.appendChild(heading);
      const scores = document.createElement('div'); scores.className = 'score-buttons';
      for (const value of ['1', '2', '3', '4', '5', ...(rubric.key === keys[3] ? ['NA'] : [])]) {
        const label = document.createElement('label'), input = document.createElement('input');
        input.type = 'radio'; input.name = rubric.key; input.dataset.score = rubric.key; input.value = value;
        label.title = value === 'NA' ? '背景本来无法比较' : rubric.anchors[Number(value) - 1];
        label.append(input, document.createTextNode(value)); scores.appendChild(label);
      } card.appendChild(scores);
      const details = document.createElement('details'), summary = document.createElement('summary'), anchors = document.createElement('ol');
      summary.textContent = '查看1–5分锚点'; anchors.className = 'anchors';
      rubric.anchors.forEach(text => { const li = document.createElement('li'); li.textContent = text; anchors.appendChild(li); });
      details.append(summary, anchors); card.appendChild(details);
      const choices = document.createElement('div'); choices.className = 'choices';
      config.issue_options[rubric.key].forEach(text => {
        const label = document.createElement('label'), input = document.createElement('input'); input.type = 'checkbox'; input.dataset.issue = rubric.key; input.value = text;
        label.className = 'choice'; label.append(input, document.createTextNode(text)); choices.appendChild(label);
      }); card.appendChild(choices); el('metrics').appendChild(card);
      const p = document.createElement('p'); p.textContent = rubric.label + '：' + rubric.anchors[0] + '（1分）至' + rubric.anchors[4] + '（5分）。'; el('instructions').appendChild(p);
    });
    for (const reason of ['', ...config.na_reasons]) { const option = document.createElement('option'); option.value = reason; option.textContent = reason ? reason.replace('（备注说明）', '') : '可不填写'; el('naReason').appendChild(option); }
    rows.forEach((row, index) => { const option = document.createElement('option'); option.value = index; el('choose').appendChild(option); });
    el('owner').textContent = '评审 ' + packet.rater + ' · 四维量表v3 · 固定96个位置';
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) { draft = normalizeImport(JSON.parse(saved)); savedRevision = draft.revision; }
      el('saveStatus').textContent = saved ? '已载入本机草稿；修改评分后自动保存，结束时请下载JSON备份。' : '选择分数后尝试自动保存；结束时请下载JSON备份。';
    } catch (_) { storageWorks = false; message('未能恢复浏览器草稿。可导入JSON备份或已有Excel；请保留原文件。'); }
    el('choose').onchange = event => show(Number(event.target.value)); el('previous').onclick = () => move(-1); el('next').onclick = () => move(1);
    el('speed').onchange = event => el('video').playbackRate = Number(event.target.value);
    for (const id of ['metrics', 'naReason']) el(id).addEventListener('change', readCurrent);
    el('backup').onclick = () => download(JSON.stringify({...draft, saved_at: new Date().toISOString()}, null, 2), 'progress_' + packet.rater + '_' + stamp() + '.json', 'application/json');
    for (const [id, final] of [['exportCurrent', false], ['exportFinal', true]]) el(id).onclick = () => { try { exportExcel(final); } catch (error) { message('导出失败：' + error.message, 'error'); } };
    el('importButton').onclick = () => el('importFile').click();
    el('importFile').onchange = async event => {
      const file = event.target.files[0]; if (!file) return;
      try {
        if (file.size > 6000000) throw Error('文件过大，请选择本人的进度JSON或评分Excel。');
        const data = /\.xlsx$/i.test(file.name) ? window.RatingWorkbook.importWorkbook(config, packet, new Uint8Array(await file.arrayBuffer())) : JSON.parse(await file.text());
        const candidate = normalizeImport(data);
        if ((Object.values(draft.records).some(touched) || Object.values(draft.declarations).some(Boolean)) && !window.confirm('将载入所选文件的评分并替换当前草稿。请先下载当前JSON备份。确认载入？')) return;
        draft = candidate; draft.revision = savedRevision; show(candidate.position); save(); message('已载入本人同批次记录；可继续评分并导出Excel。', 'success');
      } catch (error) { message('导入失败，原草稿保留：' + error.message, 'error'); }
      finally { event.target.value = ''; }
    };
    window.addEventListener('storage', event => {
      if (event.key !== storageKey && event.key !== null) return;
      try { if (event.newValue && JSON.parse(event.newValue).revision === savedRevision) return; } catch (_) { /* Preserve this tab if a stored draft is malformed. */ }
      conflict = true; save();
    });
    window.addEventListener('beforeunload', event => { if ((!storageWorks || conflict) && Object.values(draft.records).some(touched)) { event.preventDefault(); event.returnValue = ''; } });
    window.addEventListener('hashchange', () => { const next = rows.findIndex(row => row.clip_id === location.hash.slice(1)); if (next >= 0 && next !== position) show(next); });
    el('video').onloadedmetadata = refresh;
    el('video').onerror = () => { if (stateFor(rows[position].clip_id).status === 'success') el('videoError').textContent = '视频未能打开，请确认完整解压、videos相对路径正确，或单独打开视频。'; };
    const hashIndex = rows.findIndex(row => row.clip_id === location.hash.slice(1)); show(hashIndex >= 0 ? hashIndex : draft.position);
    window.RatingApp = {validateRecord, completed, materialize: clip => ({evidence_time: timeText(recordFor(clip), clip), comment: commentText(recordFor(clip), clip)})};
  } catch (error) {
    const target = el('messages'); if (target) { target.className = 'error'; target.textContent = error.message; }
    document.querySelectorAll('button').forEach(button => button.disabled = true);
  }
})();
