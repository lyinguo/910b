import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const planPath=process.argv[2],mode=process.argv[3];assert.ok(['preview','author'].includes(mode));
const plan=JSON.parse(await fs.readFile(planPath,'utf8')),home=path.dirname(planPath);
assert.equal(crypto.createHash('sha256').update(await fs.readFile(plan.source_backup)).digest('hex'),plan.expected_before_sha256);
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.source_backup));const sheet=book.worksheets.getItem(plan.sheet_name);
assert.equal(plan.changes.length,1);const contexts=plan.changes.map(c=>sheet.getRange(`B${c.row}:F${c.row}`).values);
for(const c of plan.changes){const [task,model,d]=sheet.getRange(`B${c.row}:D${c.row}`).values[0];assert.equal(model,c.model_id);assert.equal(task,c.task);assert.equal(Number(d),Number(c.A100_full_precision_cached_D));assert.ok(sheet.getRange(c.cell).values[0][0]===null||sheet.getRange(c.cell).values[0][0]==='');assert.ok(sheet.getRange(c.cell).formulas[0][0]===null||sheet.getRange(c.cell).formulas[0][0]==='');}
if(mode==='preview'){
 for(const c of plan.changes){const image=await book.render({sheetName:plan.sheet_name,range:`B${c.row-2}:G${c.row+2}`,scale:1.25,format:'png'});await fs.writeFile(path.join(home,`before_${c.row}.png`),new Uint8Array(await image.arrayBuffer()),{flag:'wx'});}
 const view=await book.inspect({kind:'table',range:`${plan.sheet_name}!B353:G357`,include:'values,formulas',tableMaxRows:5,tableMaxCols:6,maxChars:2500});await fs.writeFile(path.join(home,'original_target_view.ndjson'),view.ndjson,{flag:'wx'});console.log(JSON.stringify({preview:true,production_unchanged:true,change:plan.changes[0]}));
}else{
 for(const c of plan.changes)await fs.access(path.join(home,`before_${c.row}.png`));
 for(const c of plan.changes)sheet.getRange(c.cell).values=[[c.new_value]];
 book.recalculate();plan.changes.forEach((c,i)=>assert.deepEqual(sheet.getRange(`B${c.row}:F${c.row}`).values,contexts[i]));
 const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'unchanged-formula error scan'});await fs.writeFile(path.join(home,'formula_scan.ndjson'),errors.ndjson,{flag:'wx'});
 for(const c of plan.changes){const image=await book.render({sheetName:plan.sheet_name,range:`B${c.row-2}:G${c.row+2}`,scale:1.25,format:'png'});await fs.writeFile(path.join(home,`after_${c.row}.png`),new Uint8Array(await image.arrayBuffer()),{flag:'wx'});}
 await (await SpreadsheetFile.exportXlsx(book)).save(path.join(home,'artifact_authoring_candidate.xlsx'));
 const values=sheet.getRange(`G2:G${plan.last_model_row}`).values;for(const c of plan.changes)assert.equal(values[c.row-2][0],c.new_value);
 await fs.writeFile(path.join(home,'artifact_authored_G_values.json'),JSON.stringify(values),{flag:'wx'});console.log(JSON.stringify({authored:true,changes:plan.changes,context_preserved:true,production_unchanged:true,numeric_proof_and_idle_original_FLOCK_still_required:true}));
}
