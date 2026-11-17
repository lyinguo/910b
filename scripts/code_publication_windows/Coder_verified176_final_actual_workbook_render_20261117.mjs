import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const planPath=process.argv[2],home=path.dirname(planPath),plan=JSON.parse(await fs.readFile(planPath,'utf8'));
const bytes=await fs.readFile(plan.target);assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),'6d73108e181887c9dfd95b7c851981c1ac316238705fe37e8c5bf59c15dc9a82');
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.target)),sheet=wb.worksheets.getItem(plan.sheet_name);let values=[];
for(const c of plan.changes){const [task,mid,d]=sheet.getRange(`B${c.row}:D${c.row}`).values[0];assert.equal(task,c.task);assert.equal(mid,c.model_id);assert.equal(Number(d),Number(c.A100_full_precision_cached_D));assert.equal(sheet.getRange(c.cell).values[0][0],c.new_value);const im=await wb.render({sheetName:plan.sheet_name,range:`B${c.row-2}:G${c.row+2}`,scale:1.25,format:'png'});await fs.writeFile(path.join(home,`final_actual_${c.row}.png`),new Uint8Array(await im.arrayBuffer()),{flag:'wx'});values.push({model_id:mid,task,D:d,cell:c.cell,G:c.new_value});}
const r={actual_target:plan.target,actual_SHA256:crypto.createHash('sha256').update(bytes).digest('hex'),actual_one_G_verified:true,values,no_workbook_write_or_export:true};await fs.writeFile(path.join(home,'final_actual_workbook_view.json'),JSON.stringify(r,null,2),{flag:'wx'});console.log(JSON.stringify(r));
