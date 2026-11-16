/* Offline XLSX editing: preserve the trusted template's styles, links and validations. */
(function (global) {
  'use strict';
  const NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main';
  const REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
  const decoder = new TextDecoder(), encoder = new TextEncoder();
  function xml(bytes) {
    if (!bytes || bytes.length > 3000000) throw Error('Excel XML缺失或过大。');
    const text = decoder.decode(bytes);
    if (/<!DOCTYPE/i.test(text)) throw Error('不支持含DOCTYPE的Excel文件。');
    const doc = new DOMParser().parseFromString(text, 'application/xml');
    if (doc.getElementsByTagName('parsererror').length) throw Error('Excel XML解析失败。');
    return doc;
  }
  const elements = (node, name) => Array.from(node.getElementsByTagNameNS(NS, name));
  function unzip(bytes) {
    if (bytes.length > 6000000) throw Error('Excel过大，请选择本项目的评分表。');
    return global.fflate.unzipSync(bytes, {filter: file => file.originalSize <= 3000000 &&
      (/^xl\/(workbook\.xml|_rels\/workbook\.xml\.rels|sharedStrings\.xml|worksheets\/sheet\d+\.xml)$/.test(file.name))});
  }
  function sheets(files) {
    const book = xml(files['xl/workbook.xml']);
    const relationships = xml(files['xl/_rels/workbook.xml.rels']);
    const rels = Object.fromEntries(Array.from(relationships.documentElement.children).map(r => [r.getAttribute('Id'), r.getAttribute('Target')]));
    const names = elements(book, 'sheet');
    if (names.map(s => s.getAttribute('name')).join('|') !== '评分表|评分说明|评审声明|表单信息') throw Error('不是原评分表的四个工作表。');
    return Object.fromEntries(names.map(s => {
      const target = rels[s.getAttributeNS(REL, 'id')];
      if (!target) throw Error('工作表关系缺失。');
      const path = target.startsWith('/') ? target.slice(1) : 'xl/' + target;
      if (!/^xl\/worksheets\/sheet\d+\.xml$/.test(path)) throw Error('工作表路径不支持。');
      return [s.getAttribute('name'), {path, doc: xml(files[path])}];
    }));
  }
  function strings(files) {
    return files['xl/sharedStrings.xml'] ? elements(xml(files['xl/sharedStrings.xml']), 'si').map(si => elements(si, 't').map(t => t.textContent).join('')) : [];
  }
  function cells(doc, shared) {
    const map = new Map(elements(doc, 'c').map(c => [c.getAttribute('r'), c]));
    return {map, get(address) {
      const c = map.get(address);
      if (!c) return '';
      if (elements(c, 'f').length) throw Error(address + '：请使用字面分数或文字，不能使用公式。');
      const type = c.getAttribute('t'), value = elements(c, 'v')[0]?.textContent ?? '';
      if (type === 'inlineStr') return elements(c, 't').map(t => t.textContent).join('');
      if (type === 's') {
        const index = Number(value);
        if (!Number.isInteger(index) || shared[index] === undefined) throw Error('Excel字符串索引无效。');
        return shared[index];
      }
      if (type === 'b') return value === '1' ? true : false;
      return value;
    }};
  }
  function setCell(doc, address, value, numeric = false) {
    let cell = elements(doc, 'c').find(c => c.getAttribute('r') === address);
    if (!cell) {
      const number = address.replace(/\D/g, '');
      let row = elements(doc, 'row').find(r => r.getAttribute('r') === number);
      if (!row) throw Error('模板缺少评分行。');
      cell = doc.createElementNS(NS, 'c'); cell.setAttribute('r', address);
      const column = address.replace(/\d/g, '');
      const next = Array.from(row.children).find(c => c.getAttribute('r').replace(/\d/g, '').localeCompare(column) > 0);
      row.insertBefore(cell, next || null);
    }
    while (cell.firstChild) cell.removeChild(cell.firstChild);
    cell.removeAttribute('t');
    if (value === '' || value === null || value === undefined) return;
    if (numeric) {
      const v = doc.createElementNS(NS, 'v'); v.textContent = String(value); cell.appendChild(v);
    } else {
      cell.setAttribute('t', 'inlineStr');
      const is = doc.createElementNS(NS, 'is'), t = doc.createElementNS(NS, 't');
      t.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve');
      t.textContent = String(value); is.appendChild(t); cell.appendChild(is);
    }
  }
  function templateBytes(config) {
    return Uint8Array.from(atob(config.template_base64), c => c.charCodeAt(0));
  }
  function exportWorkbook(config, values, declarations) {
    // Export only from the distributed blank template, never from imported macros/objects.
    const files = global.fflate.unzipSync(templateBytes(config));
    const sheetsByName = sheets(files), rating = sheetsByName['评分表'], decl = sheetsByName['评审声明'];
    values.forEach((row, index) => {
      const values = [...config.scores.map(key => row[key]), row.background_na_reason, row.evidence_time, row.comment];
      values.forEach((value, col) => setCell(rating.doc, 'EFGHIJK'[col] + (index + 6), value,
        col < 4 && value !== '' && value !== 'NA'));
    });
    ['name', 'date', 'seen_model_scores', 'discussed_video_defects', 'other'].forEach((key, index) => setCell(decl.doc, 'B' + (index + 2), declarations[key] || ''));
    [rating, decl].forEach(sheet => files[sheet.path] = encoder.encode(new XMLSerializer().serializeToString(sheet.doc)));
    return global.fflate.zipSync(files, {level: 6});
  }
  function importWorkbook(config, packet, bytes) {
    const files = unzip(bytes), worksheets = sheets(files), shared = strings(files);
    const meta = cells(worksheets['表单信息'].doc, shared);
    for (const [address, expected] of [['B1', config.protocol], ['B2', config.batch_id], ['B3', packet.rater], ['B4', '96']]) {
      if (meta.get(address) !== expected) throw Error('Excel评审/批次/协议不匹配：' + address);
    }
    if (meta.get('B5') && meta.get('B5') !== config.rating_policy) throw Error('Excel评分规则版本不匹配。');
    const rating = cells(worksheets['评分表'].doc, shared);
    const records = {};
    packet.rows.forEach((item, index) => {
      const number = index + 6;
      if (rating.get('B' + number) !== item.clip_id || rating.get('A' + number) !== String(index + 1) || rating.get('L' + number) !== item.prompt)
        throw Error('Excel匿名编号、顺序或提示词已改变。');
      const scores = {};
      config.scores.forEach((key, col) => {
        const value = rating.get('EFGH'[col] + number);
        if (typeof value === 'boolean' || !['', '1', '2', '3', '4', '5', ...(col === 3 ? ['NA'] : [])].includes(value))
          throw Error(item.clip_id + '：Excel包含无效分数。');
        scores[key] = value;
      });
      const record = {scores, issues: {}, time_ranges: [], custom_start: '', custom_end: '',
        na_reason: rating.get('I' + number), manual_time: rating.get('J' + number), notes: rating.get('K' + number), confirmed_failure: false};
      if (Object.values(scores).some(Boolean) || record.na_reason || record.manual_time || record.notes) records[item.clip_id] = record;
    });
    const declaration = cells(worksheets['评审声明'].doc, shared);
    if (declaration.get('B1') !== packet.rater) throw Error('Excel声明的评审身份不一致。');
    const declarations = Object.fromEntries(['name', 'date', 'seen_model_scores', 'discussed_video_defects', 'other'].map((key, index) => [key, declaration.get('B' + (index + 2))]));
    return {version: config.version, protocol: config.protocol, batch_id: config.batch_id, rater: packet.rater, records, declarations};
  }
  global.RatingWorkbook = {exportWorkbook, importWorkbook};
})(window);
