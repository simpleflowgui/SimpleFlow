import { Handle, Position } from 'reactflow';


const guidebook = {
    "text":{"element":"input","type":"text"},
    "radio":{"element":"input","type":"radio"},
    "file":{"element":"button","type":"file"},
    "select":{"element":"select","type":"none"}
}



function OutputImageNode({ data, isConnectable }) {
    var ids = [data["id"]]
    return (
    <div id="node" className="output-image-node">
        <p id={"nodelabel_"+ids[0]} className='nodelabel'>Output Image</p>  
        <Handle className="handle" type="target" position={Position.Left} isConnectable={isConnectable} />
        <div className='cont'>
            <img id={"image_"+ids[0]} src ="" ></img>
        </div>
    </div>
    );
}

function Loop({ data, isConnectable }) {
    var ids = [data["id"]]
    return (
      <div className="custom-node-collapsed">  
        <p id={"nodelabel_"+ids[0]} className='nodelabel'>Loop</p>  
        <Handle className="handle" type="target" position={Position.Left} isConnectable={isConnectable} />
        <Handle className="handle" type="source" position={Position.Right} isConnectable={isConnectable} /> 
      </div>
      );
}


function StartNode({ data, isConnectable }) {
    var ids = [data["id"]]
    return (
      <div className="custom-node-collapsed">  
        <p id={"nodelabel_"+ids[0]} className='nodelabel'>Start</p>  
        <Handle className="handle" type="source" position={Position.Right} id="a" isConnectable={isConnectable} />
      </div>
      );
}


function EndNode({ data, isConnectable }) {
    var ids = [data["id"]]
    return (
      <div className="custom-node-collapsed">
        <p id={"nodelabel_"+ids[0]} className='nodelabel'>End</p>    
        <Handle className="handle" type="target" position={Position.Left}  isConnectable={isConnectable} />
      </div>
      );
}


function CameraVideoInput({ data, isConnectable }) {
    var ids = [data["id"]]
    var func = function(){
        function onChange(event){
            data["inps"]["camindex"]=event.target.value;  
        }
        var cont = document.createElement("div");
        cont.id = "cont_"+"node_"+ids[0];
        cont.className = "cont";
        if(!document.getElementById("cont_"+"node_"+ids[0])){
            document.getElementById("node_"+ids[0]).append(cont);
        }
        cont.style.visibility = "hidden";
        var newel = document.createElement("p");
        newel.id = "p_"+"node_"+ids[0];
        newel.className = "text";
        newel.innerHTML = "Select Camera By Index";
        if(!document.getElementById("p_node_"+ids[0])){
            cont.append(newel);
        }
        newel = document.createElement("select");
        newel.id = "Camera_Select_"+ids[0];
        newel.className = "defaultselect";
        newel.addEventListener("change",function(event){
            onChange(event);
        });
        cont.append(newel);        
        newel = document.createElement("option");
        newel.id = "select_"+"camera"+"_option_"+ids[0];
        newel.value = "select";
        newel.innerHTML = "Select";
        if(!document.getElementById("select_"+"camera"+"_option_"+ids[0])){
            document.getElementById("Camera_Select_"+ids[0]).append(newel);
        }        
    }
    intervalcreate("node_"+ids[0],func);
    if(document.getElementById("cont_"+"node_"+ids[0])){
        if(document.getElementById("cont_"+"node_"+ids[0]).style.visibility == "visible"){
            data["expandtoggle"] = false;
        }        
    }
    else{
        data["expandtoggle"] = true;
    }    
    function toggleexpand(){
        if(data["expandtoggle"]){
            if(Object.keys(data["datamethods"]).length == 0){
                if(document.getElementById("cont_"+"node_"+ids[0])){
                    //document.getElementById("cont_"+"node_"+ids[0]+"_"+0).remove();
                    document.getElementById("cont_"+"node_"+ids[0]).style.visibility = "hidden";
                }
            }
            data["expandtoggle"] = false;
            document.getElementById("nodelabel_"+ids[0]).className = "";
            document.getElementById("node_"+ids[0]).className = "custom-node-collapsed";
            document.getElementById("arrow_"+ids[0]).className = "arrow-expand";
            document.getElementById("node_"+ids[0]).style.height = "70px";

        }
        else{        
            if(document.getElementById("cont_"+"node_"+ids[0])){
                document.getElementById("cont_"+"node_"+ids[0]).style.visibility = "visible";
            }            
            data["expandtoggle"] = true;
            document.getElementById("nodelabel_"+ids[0]).className = "nodelabel";
            document.getElementById("node_"+ids[0]).className = "custom-node-expanded";
            document.getElementById("arrow_"+ids[0]).className = "arrow-collapse";
            document.getElementById("node_"+ids[0]).style.height = "auto";
        }
        
    }
    if(!data["dropped"]){
        data["funcs"].getcams(data);
        data["dropped"]=true;
    }
    var updateinps = {
        update_data: function(cams){
            if(cams.split(",").length>1){
                let cameras = cams.split(",");
                for(i=0;i<cameras.length;i++){
                    let newel = document.createElement("option");
                    newel.id = cameras[i]+"_option_"+ids[0];
                    newel.value = cameras[i];
                    newel.innerHTML = cameras[i];
                    if(!document.getElementById(cameras[i]+"_option_"+ids[0])){
                        document.getElementById("Camera_Select_"+ids[0]).append(newel);
                    }        
                }
            }
            else if(cams==""){
            }
            else{
                let newel = document.createElement("option");
                newel.id = cams+"_option_"+ids[0];
                newel.value = cams;
                newel.innerHTML = cams;
                if(!document.getElementById(cams+"_option_"+ids[0])){
                    document.getElementById("Camera_Select_"+ids[0]).append(newel);
                }    
            }
        }
    }
    data["update"] = updateinps;
    return (
    <div id={"node_"+ids[0]} className="custom-node-collapsed">
        <p id={"nodelabel_"+ids[0]} className='nodelabel'>Camera Video Input</p>  
        <Handle className="handle" type="target" position={Position.Left} isConnectable={isConnectable} />
        <Handle className="handle" type="source" position={Position.Right} isConnectable={isConnectable} /> 
        <div id={"expand_"+ids[0]} className='expand' onClick={toggleexpand}>
            <div id={"arrow_"+ids[0]} className='arrow-expand'></div>
        </div>

    </div>
        );
}


function SelectCamera({ data, isConnectable }) {
    var ids = [data["id"]]
    var func = function(){
        function onChange(event){
            data["inps"]["camindex"]=event.target.value;  
        }
        var cont = document.createElement("div");
        cont.id = "cont_"+"node_"+ids[0];
        cont.className = "cont";
        if(!document.getElementById("cont_"+"node_"+ids[0])){
            document.getElementById("node_"+ids[0]).append(cont);
        }
        cont.style.visibility = "hidden";
        var newel = document.createElement("p");
        newel.id = "p_"+"node_"+ids[0];
        newel.className = "text";
        newel.innerHTML = "Select Camera By Index";
        if(!document.getElementById("p_node_"+ids[0])){
            cont.append(newel);
        }
        newel = document.createElement("select");
        newel.id = "Camera_Select_"+ids[0];
        newel.className = "defaultselect";
        newel.addEventListener("change",function(event){
            onChange(event);
        });
        cont.append(newel);        
        newel = document.createElement("option");
        newel.id = "select_"+"camera"+"_option_"+ids[0];
        newel.value = "select";
        newel.innerHTML = "Select";
        if(!document.getElementById("select_"+"camera"+"_option_"+ids[0])){
            document.getElementById("Camera_Select_"+ids[0]).append(newel);
        }        
    }
    intervalcreate("node_"+ids[0],func);
    if(document.getElementById("cont_"+"node_"+ids[0])){
        if(document.getElementById("cont_"+"node_"+ids[0]).style.visibility == "visible"){
            data["expandtoggle"] = false;
        }        
    }
    else{
        data["expandtoggle"] = true;
    }    
    function toggleexpand(){
        if(data["expandtoggle"]){
            if(Object.keys(data["datamethods"]).length == 0){
                if(document.getElementById("cont_"+"node_"+ids[0])){
                    //document.getElementById("cont_"+"node_"+ids[0]+"_"+0).remove();
                    document.getElementById("cont_"+"node_"+ids[0]).style.visibility = "hidden";
                }
            }
            data["expandtoggle"] = false;
            document.getElementById("nodelabel_"+ids[0]).className = "";
            document.getElementById("node_"+ids[0]).className = "custom-node-collapsed";
            document.getElementById("arrow_"+ids[0]).className = "arrow-expand";
            document.getElementById("node_"+ids[0]).style.height = "70px";

        }
        else{        
            if(document.getElementById("cont_"+"node_"+ids[0])){
                document.getElementById("cont_"+"node_"+ids[0]).style.visibility = "visible";
            }            
            data["expandtoggle"] = true;
            document.getElementById("nodelabel_"+ids[0]).className = "nodelabel";
            document.getElementById("node_"+ids[0]).className = "custom-node-expanded";
            document.getElementById("arrow_"+ids[0]).className = "arrow-collapse";
            document.getElementById("node_"+ids[0]).style.height = "auto";
        }
        
    }
    if(!data["dropped"]){
        data["funcs"].getcams(data);
        data["dropped"]=true;
    }
    var updateinps = {
        update_data: function(cams){
            if(cams.split(",").length>1){
                let cameras = cams.split(",");
                for(i=0;i<cameras.length;i++){
                    let newel = document.createElement("option");
                    newel.id = cameras[i]+"_option_"+ids[0];
                    newel.value = cameras[i];
                    newel.innerHTML = cameras[i];
                    if(!document.getElementById(cameras[i]+"_option_"+ids[0])){
                        document.getElementById("Camera_Select_"+ids[0]).append(newel);
                    }        
                }
            }
            else if(cams==""){
            }
            else{
                let newel = document.createElement("option");
                newel.id = cams+"_option_"+ids[0];
                newel.value = cams;
                newel.innerHTML = cams;
                if(!document.getElementById(cams+"_option_"+ids[0])){
                    document.getElementById("Camera_Select_"+ids[0]).append(newel);
                }    
            }
        }
    }
    data["update"] = updateinps;
    return (
    <div id={"node_"+ids[0]} className="custom-node-collapsed">
        <p id={"nodelabel_"+ids[0]} className='nodelabel'>Select Camera</p>  
        <Handle className="handle" type="target" position={Position.Left} isConnectable={isConnectable} />
        <Handle className="handle" type="source" position={Position.Right} isConnectable={isConnectable} /> 
        <div id={"expand_"+ids[0]} className='expand' onClick={toggleexpand}>
            <div id={"arrow_"+ids[0]} className='arrow-expand'></div>
        </div>

    </div>
        );
}

function intervalcreate(elem,func){
    if(document.getElementById(elem)){
        func();
    }
    else{
        setTimeout(() => {
            intervalcreate(elem,func)
        }, 100);
    }
}



function CustomNode({ data, isConnectable }) {
    var ids = [data["id"]];
    if(document.getElementById("cont_"+"node_"+ids[0]+"_"+0)){
        if(document.getElementById("cont_"+"node_"+ids[0]+"_"+0).style.visibility == "visible"){
            data["expandtoggle"] = false;
        }        
    }
    else{
        data["expandtoggle"] = true;
    }    
    function addelements(){
        let clr = JSON.parse(data["color"])[data["innerHTML"]];
        if(data["datamethods"] && data["datamethods"]!=""){    
            for(let i = 0;i<Object.keys(data["datamethods"]).length;i++){
                function onChange(event){
                    data["inps"]["vars"][data["datamethods"][i][1]]=event.target.value;  
                    data["varstest"] = event.target.value;
                }
                if(!document.getElementById("cont_"+"node_"+ids[0]+"_"+i)){
                    let func = function(){
                        var cont = document.createElement("div");
                        cont.id = "cont_"+"node_"+ids[0]+"_"+i;
                        cont.className = "cont";
                        if(!document.getElementById("cont_"+"node_"+ids[0]+"_"+i)){
                            document.getElementById("node_"+ids[0]).append(cont);
                        }
                        var newel = document.createElement("p");
                        newel.id = "p_"+"node_"+ids[0]+"_"+i;
                        newel.className = "text";
                        newel.innerHTML = data["datamethods"][i][1];
                        if(!document.getElementById("p_node_"+ids[0]+"_"+i)){
                            cont.append(newel);
                        }
                        if(guidebook[data["datamethods"][i][0]]["element"]=="button"){
                            newel = document.createElement(guidebook[data["datamethods"][i][0]]["element"]);
                            let func = function(){
                                try{
                                    var obj = JSON.parse(data["datamethods"][i][2]);
                                    if(data["datamethods"][i].length<4){
                                        document.getElementById(data["datamethods"][i][1]+"_"+ids[0]).disabled = true;
                                    }
                                    else{
                                        if(data["datamethods"][i][3]!=""){
                                            document.getElementById(data["datamethods"][i][1]+"_"+ids[0]).disabled = false;
                                        }
                                    }
                                    fileUploaded(data["datamethods"][i][3],data["datamethods"][i][1]);
                                }
                                catch(error){
                                    fileUploaded(data["datamethods"][i][2],data["datamethods"][i][1]);
                                }
                                
                            }
                            newel.addEventListener("click",func);
                            newel.innerHTML = "UPLOAD";
                            newel.id = data["datamethods"][i][1]+"_"+ids[0];
                            newel.style.borderRadius = "5%";
                            try{
                                var obj = JSON.parse(data["datamethods"][i][2]);
                                newel.disabled = true;
                            }
                            catch(error){
                                newel.disabled = false;
                            }
                            if(!document.getElementById(data["datamethods"][i][1]+"_"+ids[0])){
                                cont.append(newel);
                            }
                        }
                        else if(guidebook[data["datamethods"][i][0]]["element"]=="select"){
                            newel = document.createElement("select");
                            newel.id = data["datamethods"][i][1]+"_"+ids[0];
                            newel.className = "defaultselect";
                            if(data["datamethods"][i].length<=3){
                                newel.addEventListener("change",function(event){
                                    onChange(event);
                                });
                            }
                            else{
                                newel.addEventListener("change",function(event){
                                    dropdowndynamic(event,newel.id,i);
                                });
                            }
                            cont.append(newel);
                            if(Array.isArray(data["datamethods"][i][2])){
                                newel = document.createElement("option");
                                newel.id = "select_"+data["datamethods"][i][2]+"_option_"+ids[0];
                                newel.value = "select";
                                newel.innerHTML = "Select";
                                if(!document.getElementById("select_"+data["datamethods"][i][2]+"_option_"+ids[0])){
                                    document.getElementById(data["datamethods"][i][1]+"_"+ids[0]).append(newel);
                                }
                                for(let j=0;j<data["datamethods"][i][2].length;j++){
                                    newel = document.createElement("option");
                                    newel.id = data["datamethods"][i][2][j]+"_option_"+ids[0];
                                    newel.value = data["datamethods"][i][2][j];
                                    newel.innerHTML = data["datamethods"][i][2][j];
                                    if(!document.getElementById(data["datamethods"][i][2][j]+"_option_"+ids[0])){
                                        document.getElementById(data["datamethods"][i][1]+"_"+ids[0]).append(newel);
                                    }
                                }
                            }
                            else{

                                let obj = JSON.parse(data["datamethods"][i][2]);
                                let newel = document.createElement("option");
                                newel.id = "select_"+Object.keys(obj)[0]+"_option_"+ids[0];
                                newel.value = "select";
                                newel.innerHTML = "Select";
                                if(!document.getElementById("select_"+Object.keys(obj)[0]+"_option_"+ids[0])){
                                    document.getElementById(data["datamethods"][i][1]+"_"+ids[0]).append(newel);
                                }
                                for(let j=0;j<obj[Object.keys(obj)[0]].length;j++){
                                    let newel = document.createElement("option");
                                    newel.id = obj[Object.keys(obj)[0]][j]+"_option_"+ids[0];
                                    newel.value = obj[Object.keys(obj)[0]][j];
                                    newel.innerHTML = obj[Object.keys(obj)[0]][j];
                                    if(!document.getElementById(obj[Object.keys(obj)[0]][j]+"_option_"+ids[0])){
                                        document.getElementById(data["datamethods"][i][1]+"_"+ids[0]).append(newel);
                                    }
                                }
                            }
                        }
                        else{
                            if(data["datamethods"][i][2] && data["datamethods"][i][2]!="File" && data["datamethods"][i][2]!="Folder"){
                                cont.className ="cont";
                                for(let j=0;j<data["datamethods"][i][2].length;j++){
                                    newel = document.createElement(guidebook[data["datamethods"][i][0]]["element"]);
                                    newel.type = guidebook[data["datamethods"][i][0]]["type"]
                                    newel.id = data["datamethods"][i][2][j]+"_"+ids[0];
                                    newel.name = "choice_"+i+"_"+ids[0];
                                    newel.value = data["datamethods"][i][2][j];
                                    newel.className = "nodrag";
                                    if(data["datamethods"][i].length<=3){
                                        newel.addEventListener("change",function(event){
                                            onChange(event);
                                        });
                                    }
                                    else{
                                        newel.addEventListener("change",function(event){
                                            dropdowndynamic(event,newel.id,i);
                                        });
                                    }
                                    if(!document.getElementById(data["datamethods"][i][2][j]+"_"+ids[0])){
                                        cont.append(newel);
                                    }
                                    newel = document.createElement("label");
                                    newel.style.display = "inline-block";
                                    newel.style.width = "90%";
                                    newel.innerHTML = data["datamethods"][i][2][j];
                                    newel.id = "label_"+data["datamethods"][i][2][j]+"_"+ids[0];
                                    newel.htmlFor = data["datamethods"][i][2][j]+"_"+ids[0];
                                    if(!document.getElementById("label_"+data["datamethods"][i][2][j]+"_"+ids[0])){
                                        cont.append(newel);
                                    }
                                    
                                }
                            }
                            else{
                                newel = document.createElement(guidebook[data["datamethods"][i][0]]["element"]);
                                newel.id = guidebook[data["datamethods"][i][0]]["element"]+"_"+"node_"+ids[0]+"_"+i;
                                newel.type = guidebook[data["datamethods"][i][0]]["type"]
                                newel.addEventListener("change",function(event){
                                    onChange(event);
                                });
                                if(!document.getElementById(guidebook[data["datamethods"][i][0]]["element"]+"_"+"node_"+ids[0]+"_"+i)){
                                    cont.append(newel);
                                }
                            }
                        }
                    };
                    intervalcreate("node_"+ids[0],func);  
                } 

            }            
        }
    }
    function toggleexpand(){
        if(data["expandtoggle"]){
            if(Object.keys(data["datamethods"]).length == 0){
                if(document.getElementById("cont_"+"node_"+ids[0]+"_"+0)){
                    //document.getElementById("cont_"+"node_"+ids[0]+"_"+0).remove();
                    document.getElementById("cont_"+"node_"+ids[0]+"_"+0).style.visibility = "hidden";
                }
            }
            else{
                for(let i =0;i< Object.keys(data["datamethods"]).length;i++){
                    if(document.getElementById("cont_"+"node_"+ids[0]+"_"+i)){
                        document.getElementById("cont_"+"node_"+ids[0]+"_"+i).style.visibility = "hidden";
                        //document.getElementById("cont_"+"node_"+ids[0]+"_"+i).remove();
                    }
                }
            }
            data["expandtoggle"] = false;
            document.getElementById("nodelabel_"+ids[0]).className = "";
            document.getElementById("node_"+ids[0]).className = "custom-node-collapsed";
            document.getElementById("arrow_"+ids[0]).className = "arrow-expand";
            document.getElementById("node_"+ids[0]).style.height = "70px";

        }
        else{
            for(let i =0;i< Object.keys(data["datamethods"]).length;i++){
                if(document.getElementById("cont_"+"node_"+ids[0]+"_"+i)){
                    document.getElementById("cont_"+"node_"+ids[0]+"_"+i).style.visibility = "visible";
                }
            }
            
            data["expandtoggle"] = true;
            document.getElementById("nodelabel_"+ids[0]).className = "nodelabel";
            document.getElementById("node_"+ids[0]).className = "custom-node-expanded";
            document.getElementById("arrow_"+ids[0]).className = "arrow-collapse";
            document.getElementById("node_"+ids[0]).style.height = "auto";
        }
        
    }
    function dropdowndynamic(evt,d,b){
        data["inps"]["vars"][String(d).split("_")[0]]=evt.target.value; 
        if(Array.isArray(data["datamethods"][b][3])==false){    
            if(document.getElementById(data["datamethods"][b][3]+"_"+ids[0])){
                for(let n=0;n<Object.keys(data["datamethods"]).length;n++){
                    if(data["datamethods"][n][1]==data["datamethods"][b][3]){
                        if(data["datamethods"][n][0]=="select"){                        
                            let childs = document.getElementById(data["datamethods"][n][1]+"_"+ids[0]);
                            while (childs.firstChild){                   
                                childs.removeChild(childs.lastChild);                    
                            }
                            let obj = JSON.parse(data["datamethods"][n][2]);
                            let newel = document.createElement("option");
                            newel.id = "select_"+evt.target.value+"_option_"+ids[0];
                            newel.value = "select";
                            newel.innerHTML = "Select";
                            if(!document.getElementById("select_"+evt.target.value+"_option_"+ids[0])){
                                document.getElementById(data["datamethods"][b][3]+"_"+ids[0]).append(newel);
                            }
                            for(let j=0;j<obj[evt.target.value].length;j++){
                                let newel = document.createElement("option");
                                newel.id = obj[evt.target.value][j]+"_option_"+ids[0];
                                newel.value = obj[evt.target.value][j];
                                newel.innerHTML = obj[evt.target.value][j];
                                if(!document.getElementById(obj[evt.target.value][j]+"_option_"+ids[0])){
                                    document.getElementById(data["datamethods"][b][3]+"_"+ids[0]).append(newel);
                                }
                            }
                        }
                        else if(data["datamethods"][n][0]=="file"){
                            let obj = JSON.parse(data["datamethods"][n][2]);
                            data["datamethods"][n][3] = obj[evt.target.value];  
                            document.getElementById(data["datamethods"][n][1]+"_"+ids[0]).disabled = false;            
                        }
                    }
                }
            }
        }
        else{
            let els = data["datamethods"][b][3];
            for(let n1=0;n1<els.length;n1++){
                if(document.getElementById(data["datamethods"][b][3][n1]+"_"+ids[0])){
                    for(let n=0;n<Object.keys(data["datamethods"]).length;n++){
                        if(data["datamethods"][n][1]==els[n1]){
                            if(data["datamethods"][n][0]=="select"){                        
                                let childs = document.getElementById(data["datamethods"][n][1]+"_"+ids[0]);
                                while (childs.firstChild){                
                                    childs.removeChild(childs.lastChild);                    
                                }
                                let obj = JSON.parse(data["datamethods"][n][2]);
                                let newel = document.createElement("option");
                                newel.id = "select_"+evt.target.value+"_option_"+ids[0];
                                newel.value = "select";
                                newel.innerHTML = "Select";
                                if(!document.getElementById("select_"+evt.target.value+"_option_"+ids[0])){
                                    document.getElementById(els[n1]+"_"+ids[0]).append(newel);
                                }
                                for(let j=0;j<obj[evt.target.value].length;j++){
                                    let newel = document.createElement("option");
                                    newel.id = obj[evt.target.value][j]+"_option_"+ids[0];
                                    newel.value = obj[evt.target.value][j];
                                    newel.innerHTML = obj[evt.target.value][j];
                                    if(!document.getElementById(obj[evt.target.value][j]+"_option_"+ids[0])){
                                        document.getElementById(els[n1]+"_"+ids[0]).append(newel);
                                    }
                                }
                            }
                            else if(data["datamethods"][n][0]=="file"){
                                let obj = JSON.parse(data["datamethods"][n][2]);
                                data["datamethods"][n][3] = obj[evt.target.value];  
                                document.getElementById(data["datamethods"][n][1]+"_"+ids[0]).disabled = false;       
                            }
                        }
                    }
                }
            }
        }
        
    }
    function fileUploaded(f,n){
        data["inps"]["upload_type"] = f;
        data["inps"]["vars"][n] = data["upload"].uploadfile(data,n);
        
    }
    var updateinps = {
        update_data: function(path,sortcut,n){
            data["inps"]["vars"][n] = path;
            if(path != "()"){
                document.getElementById(n+"_"+ids[0]).innerHTML = sortcut;
            }
            else{
                document.getElementById(n+"_"+ids[0]).innerHTML = "Upload File";
            }
            
        }
    }
    data["update"] = updateinps;
    data["custom_node"] = {};
    data["custom_node"]["name"] = data["innerHTML"].replaceAll(" ","_");
    data["custom_node"]["type"] = data["junctions"];
    data["inps"]["custom_node"] = data["custom_node"]["name"];
    data["inps"]["lib"] = data["lib"];
    try{
        data["datamethods"] = JSON.parse(data["datamethods"]);
    }
    catch(err){

    }
    var func = function(){
        if(document.getElementById("nodelabel_"+ids[0]).innerHTML != data["innerHTML"]){
            document.getElementById("nodelabel_"+ids[0]).innerHTML = data["inps"]["custom_node"].replaceAll("_"," ");
            if(data["inps"]["custom_node"].replaceAll("_"," ").length>30){
                document.getElementById("nodelabel_"+ids[0]).style.fontSize = "12px";
                document.getElementById("nodelabel_"+ids[0]).innerHTML = data["inps"]["custom_node"].replaceAll("_"," ") + "<br/>";
            }
            else if(data["inps"]["custom_node"].replaceAll("_"," ").length<=30 && data["inps"]["custom_node"].replaceAll("_"," ").length>24){
                document.getElementById("nodelabel_"+ids[0]).style.fontSize = "12px";
                document.getElementById("nodelabel_"+ids[0]).innerHTML = data["inps"]["custom_node"].replaceAll("_"," ") + "<br/><br/>";
            }
        }
        let clr = JSON.parse(data["color"])[data["innerHTML"]];
        document.getElementById("node_"+ids[0]).style.background = "radial-gradient("+clr+"F9,"+clr+"C9)";
    }
    intervalcreate("nodelabel_"+ids[0],func);
    if(!document.getElementById("cont_"+"node_"+ids[0]+"_"+0)){
        addelements();
        let func = function(){
            if(Object.keys(data["datamethods"]).length == 0){
                document.getElementById("node_"+ids[0]).className = "custom-node-collapsed";
                document.getElementById("expand_"+ids[0]).remove();
            }
        }
        intervalcreate("expand_"+ids[0],func);  
        data["exist"] = true;
    }
    func = function(){
        let clr = JSON.parse(data["color"])[data["innerHTML"]];
        document.getElementById("expand_"+ids[0]).style.backgroundColor= clr;
        document.getElementById("expand_"+ids[0]).style.borderBottomLeftRadius = "5%";
        document.getElementById("expand_"+ids[0]).style.borderBottomRightRadius = "5%";
    }
    intervalcreate("expand_"+ids[0],func);
    if(data["custom_node"]["type"] =="targetonly"){
        return (
          <div id={"node_"+ids[0]} className="measure-node">
              <p id={"nodelabel_"+ids[0]}></p>
              <Handle className="handle" type="target" position={Position.Left} isConnectable={isConnectable} />
          </div>
          );
    }
    else if(data["custom_node"]["type"] =="sourceonly"){
    return (
        <div id={"node_"+ids[0]} className="measure-node">
            <p id={"nodelabel_"+ids[0]}>hello</p>
            <Handle className="handle" type="source" position={Position.Right} isConnectable={isConnectable} />
        </div>
        );
    }
    else{
    return (
        <div id={"node_"+ids[0]} className="custom-node-expanded">
            <p id={"nodelabel_"+ids[0]} className='nodelabel'></p>
            <Handle className="handle"  type="target" position={Position.Left} isConnectable={isConnectable} />
            <Handle className="handle"  type="source" position={Position.Right} isConnectable={isConnectable} />
            <div id={"expand_"+ids[0]} className='expand' onClick={toggleexpand}>
                <div id={"arrow_"+ids[0]} className='arrow-collapse'></div>
            </div>
        </div>
        );
    }
      
}




export default OutputImageNode;
export{
    SelectCamera,
    CameraVideoInput,
    StartNode,
    EndNode,
    CustomNode,
    Loop
}
