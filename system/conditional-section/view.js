const configPath = input.config ?? "PROJECT_CONFIG";
const sectionId = input.id;
const sourcePath = input.file;
const targetPath = dv.current().file.path;

const startMarker = `<!-- conditional-section:${sectionId}:start -->`;
const endMarker = `<!-- conditional-section:${sectionId}:end -->`;

function findRegion(text) {
    const start = text.indexOf(startMarker);
    const end = text.indexOf(endMarker, start + startMarker.length);

    if (start === -1 || end === -1 || end < start) return null;

    const raw = text.slice(start + startMarker.length, end);
    const content = raw
        .replace(/^(?:\r?\n)+/, "")
        .replace(/(?:\r?\n)+$/, "");

    return { start, end: end + endMarker.length, content };
}

function replaceRegion(text, content) {
    const region = findRegion(text);
    if (!region) return text;

    const eol = text.includes("\r\n") ? "\r\n" : "\n";
    const normalized = content
        .replace(/\r\n/g, "\n")
        .replace(/\n/g, eol)
        .replace(/^(?:\r?\n)+/, "")
        .replace(/(?:\r?\n)+$/, "");

    const block = normalized.length > 0
        ? `${startMarker}${eol}${eol}${normalized}${eol}${eol}${endMarker}`
        : `${startMarker}${eol}${endMarker}`;

    return text.slice(0, region.start) + block + text.slice(region.end);
}

async function updateFile(file, updater) {
    if (typeof app.vault.process === "function") {
        return await app.vault.process(file, updater);
    }

    const current = await app.vault.read(file);
    const updated = updater(current);
    if (updated !== current) await app.vault.modify(file, updated);
}

const config = dv.page(configPath);
const targetFile = app.vault.getAbstractFileByPath(targetPath);
const sourceFile = app.vault.getAbstractFileByPath(sourcePath);

if (!sectionId || !sourcePath || !input.task) {
    dv.paragraph("⚠️ Условный раздел настроен не полностью.");
} else if (!config) {
    dv.paragraph(`⚠️ Не найден файл конфигурации [[${configPath}]].`);
} else if (!targetFile || !sourceFile) {
    dv.paragraph("⚠️ Не найден основной файл или файл условного раздела.");
} else {
    const task = config.file.tasks.find(item => item.blockId === input.task);
    const enabled = task?.completed === true;
    const targetText = await app.vault.read(targetFile);
    const region = findRegion(targetText);

    if (!region) {
        dv.paragraph(`⚠️ Не найдены маркеры условного раздела \`${sectionId}\`.`);
    } else if (enabled && region.content.length === 0) {
        const storedContent = await app.vault.read(sourceFile);
        await updateFile(targetFile, current => replaceRegion(current, storedContent));
    } else if (!enabled && region.content.length > 0) {
        const contentToStore = region.content.replace(/(?:\r?\n)+$/, "") + "\n";

        // Сначала сохраняем содержимое, затем убираем его из основного файла:
        // при сбое возможна только временная копия, но не потеря текста.
        await updateFile(sourceFile, () => contentToStore);
        await updateFile(targetFile, current => replaceRegion(current, ""));
    }
}
