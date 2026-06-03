const vscode = require('vscode');
const http = require('http');

async function callFlaskAPI(endpoint, payload) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify(payload);
        const options = {
            hostname: '127.0.0.1',
            port: 5000,
            path: `/api/${endpoint}`,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': data.length
            },
            timeout: 5000
        };

        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', (chunk) => body += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    resolve(JSON.parse(body));
                } else {
                    reject(new Error(`Server error: Status ${res.statusCode}`));
                }
            });
        });

        req.on('error', (err) => reject(new Error(`Flask server unreachable: ${err.message}`)));
        req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request timed out'));
        });
        req.write(data);
        req.end();
    });
}

async function completeLineCommand() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const position = editor.selection.active;
    const currentLine = editor.document.lineAt(position.line).text;
    const prefix = currentLine.slice(0, position.character);

    if (!prefix.trim()) return;

    vscode.window.setStatusBarMessage('RNN consultando a Flask...', 2000);

    try {
        const response = await callFlaskAPI('complete', { prefix: prefix, max_new: 50, temperature: 0.7 });
        if (response.ok && response.suffix) {
            await editor.edit((editBuilder) => {
                editBuilder.insert(position, response.suffix);
            });
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Error de Autocompletado: ${error.message}`);
    }
}

async function suggestCommand() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const position = editor.selection.active;
    const currentLine = editor.document.lineAt(position.line).text;
    const prefix = currentLine.slice(0, position.character);

    try {
        const response = await callFlaskAPI('suggest', { prefix: prefix, n: 5 });
        if (response.ok && response.suggestions && response.suggestions.length > 0) {
            const pick = await vscode.window.showQuickPick(response.suggestions, {
                placeHolder: 'Selecciona una prediccion estructurada por tu RNN:'
            });
            if (pick) {
                const remainingText = pick.slice(prefix.length);
                await editor.edit((editBuilder) => {
                    editBuilder.insert(position, remainingText);
                });
            }
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Error al obtener sugerencias: ${error.message}`);
    }
}

function activate(context) {
    console.log('La extension de Autocompletado RNN (Flask) esta activa.');
    let completeSub = vscode.commands.registerCommand('rnnKeras.complete', completeLineCommand);
    let suggestSub = vscode.commands.registerCommand('rnnKeras.suggest', suggestCommand);
    context.subscriptions.push(completeSub, suggestSub);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};  