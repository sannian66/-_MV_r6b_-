import {chromium} from 'playwright';
const [,,html,t,out]=process.argv;
const b=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--allow-file-access-from-files']});
const p=await b.newPage({viewport:{width:1920,height:1080}});
p.on('console',m=>console.log('page:',m.text()));p.on('pageerror',e=>console.log('ERR',e.message));
await p.goto('file://'+process.cwd()+'/'+html);
await p.waitForFunction('window.ready===true',null,{timeout:60000});
await p.evaluate(`window.draw(${t})`);
await p.screenshot({path:out});await b.close();
