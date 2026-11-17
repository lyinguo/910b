import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const planPath=process.argv[2];const plan=JSON.parse(await fs.readFile(planPath,'utf8'));const home=path.dirname(planPath);
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.source_backup));const sheet=book.worksheets.getItem(plan.sheet_name);
assert.equal(plan.changes.length,1);
const contexts=plan.changes.map(c=>sheet.getRange(`B${c.row}:F${c.row}`).values);
for(const c of plan.changes){const [task,model,d]=sheet.getRange(`B${c.row}:D${c.row}`).values[0];assert.equal(model,c.model_id);assert.equal(task,c.task);assert.equal(Number(d),Number(c.A100_full_precision_cached_D));const cell=sheet.getRange(c.cell);assert.ok(cell.values[0][0]===null||cell.values[0][0]==='');assert.ok(cell.formulas[0][0]===null||cell.formulas[0][0]==='');}
for(const c of plan.changes){const im=await book.render({sheetName:plan.sheet_name,range:`B${c.row-2}:G${c.row+2}`,scale:1.25,format:'png'});await fs.writeFile(path.join(home,`before_${c.row}.png`),new Uint8Array(await im.arrayBuffer()),{flag:'wx'});}
for(const c of plan.changes)sheet.getRange(c.cell).values=[[c.new_value]];
book.recalculate();plan.changes.forEach((c,i)=>assert.deepEqual(sheet.getRange(`B${c.row}:F${c.row}`).values,contexts[i]));
const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'unchanged-formula error scan'});
await fs.writeFile(path.join(home,'formula_scan.ndjson'),errors.ndjson);
for(const c of plan.changes){const im=await book.render({sheetName:plan.sheet_name,range:`B${c.row-2}:G${c.row+2}`,scale:1.25,format:'png'});await fs.writeFile(path.join(home,`after_${c.row}.png`),new Uint8Array(await im.arrayBuffer()),{flag:'wx'});}
await (await SpreadsheetFile.exportXlsx(book)).save(path.join(home,'artifact_authoring_candidate.xlsx'));
const values=sheet.getRange('G2:G701').values;for(const c of plan.changes)assert.equal(values[c.row-2][0],c.new_value);
await fs.writeFile(path.join(home,'artifact_authored_G_values.json'),JSON.stringify(values),{flag:'wx'});
console.log(JSON.stringify({authored:true,changes:plan.changes,context_preserved:true}));
