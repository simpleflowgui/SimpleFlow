import React from 'react';

/*
window.onload = function(){
  if(document.getElementById("aside")){
    spawn();
  }
  else{
    setTimeout(() => {
      spawn();
    }, 100);
  }
  //console.log(document.getElementById("aside").id);
};

function spawn(){
  var el = document.createElement("div");
  el.id = "rand";
  el.style.top = "30px";
  el.innerHTML = "TESTING";
  el.className = "dndnode output";
  el.onDragStart = (event) => onDragStart(event, 'Start');
  el.setAttribute("draggable",true);
  document.getElementById("aside").appendChild(el);
}*/
var drawerlvl = 0;
var drawerout = false;
var flag = true;
var els = {"Basic":
      
          [["Start","Start"],
          ["OutImage","Output Image"],
          ["End","End"]],

          "Image Processing":

          [["Measure2","Measure 2"],
          ["ImageResize","Image Resize"],
          ["ImageFlip","Image Flip"],
          ["Measure","Measure"]],
          
          "Machine Learning":

          [["InputModel","Input Model"],
          ["TorchClassify","Torch Classify"],
          ["SelectModel","Select Model"],
          ["TrainTorchClassify","Train Torch Classify"],
          ["TorchDetect","Torch Detect"],
          ["TrainTorchDetect","Train Torch Detect"],
          ["TrainYOLO","Train YOLO"],
          ["PredictYOLO","Predict YOLO"],
          ["CameraVideoInput","Camera Video Input"],
          ["TestIfItWorksNode","Test Achieved"]]
};

var selectedlibrary = "";
function slide(){
  if(drawerout){
    if(drawerlvl==1){
      document.getElementById("aside").offsetHeight;
      document.getElementById("aside").style.animation = "slideinp1 1s forwards";
    }
    else if(drawerlvl==2){
      document.getElementById("aside").offsetHeight;
      document.getElementById("aside").style.animation = "slideinpall 1s forwards";
      document.getElementById(selectedlibrary).style.animation = "librarydeselect 1s forwards";
      document.getElementById(selectedlibrary+"ribbon").style.animation = "librarydeselect 1s forwards";
      selectedlibrary = "";
    }
    document.getElementById("exec").style.animation = "slideoutbuttongreen 1s forwards";
    document.getElementById("stop").style.animation = "slideoutbuttonred 1s forwards";
    document.getElementById("sf").style.animation = "slideoutbuttonsf 1s forwards";
    document.getElementById("lf").style.animation = "slideoutbuttonlf 1s forwards";
    document.getElementById("log").style.animation = "slideoutbuttonlog 1s forwards";
    drawerout = !drawerout;
    drawerlvl=0;
  }
  else{
    document.getElementById("aside").style.animation = "slideoutp1 1s forwards";
    document.getElementById("exec").style.animation = "slideinbuttongreen 1s forwards";
    document.getElementById("stop").style.animation = "slideinbuttonred 1s forwards";
    document.getElementById("sf").style.animation = "slideinbuttonsf 1s forwards";
    document.getElementById("lf").style.animation = "slideinbuttonlf 1s forwards";
    document.getElementById("log").style.animation = "slideinbuttonlog 1s forwards";
    drawerout = !drawerout;
    drawerlvl=1;
    console.log(drawerlvl);
  }
}


export default () => {

  /*const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };
  function spawn(){
    var el = document.createElement("p");
    el.innerHTML = "LIBRARIES";
    el.style.position = "absolute";
    el.style.left = "25%";
    el.style.top = "2%";
    el.style.color = "white";
    el.style.fontSize = "40px";
    el.style.fontFamily = "Verdana, Geneva, Tahoma, sans-serif";
    document.getElementById("aside").append(el);
    var keys = Object.keys(els);
    for(let i = 0;i<keys.length;i++){
      var el = document.createElement("div");
      el.className = "libraryribbon";
      el.style.left = "5%";
      el.style.top = 15+5*i+"%";
      el.id = keys[i]+"ribbon";
      document.getElementById("aside").append(el);
      var el = document.createElement("div");
      el.id = keys[i];
      el.className = "library";
      el.style.left = "5%";
      el.style.top = 15+5*i+"%";
      el.addEventListener("click",function(){
        if(selectedlibrary !=this.id){
          this.style.animation = "libraryselect 1s forwards";
          document.getElementById(this.id+"ribbon").style.animation = "libraryselect 1s forwards";
          if(selectedlibrary!=""){
            document.getElementById(selectedlibrary).style.animation = "librarydeselect 1s forwards";
            document.getElementById(selectedlibrary+"ribbon").style.animation = "librarydeselect 1s forwards";
          }
        }
        selectedlibrary = keys[i];
        let childs = document.getElementById("sideextend");
        while (childs.firstChild){
          childs.removeChild(childs.lastChild);
        }
        document.getElementById("aside").style.animation = "slideoutp2 1s forwards";
        drawerlvl=2;
        for(let j = 0;j<els[keys[i]].length;j++){
          const foo = (event) => onDragStart(event, els[keys[i]][j][0]);
          var el = document.createElement("div");
          el.innerHTML = els[keys[i]][j][1];
          el.className = "nodesv2";
          el.addEventListener('dragstart', foo)
          el.setAttribute("draggable",true);
          document.getElementById("sideextend").append(el);
        }        
      });
      document.getElementById("aside").append(el);
      var el2 = document.createElement("p");
      el2.innerHTML = keys[i];
      el.append(el2);
    }
    var el = document.createElement("div");
    el.id = "sideextend";
    document.getElementById("aside").append(el);
  };

  function addels(){
    if(document.getElementById("aside")){
      spawn();
      flag = false;
    }
    else{
      setTimeout(() => {
        addels();
      }, 100);
    }
  }
  if(flag){
    setTimeout(() => {
      if(flag){
        addels();
      }      
    }, 100);
    
  }*/
  


  return (
    <aside id="aside">
      <div className="description">You can drag these nodes to the pane on the right.</div>
    </aside>
  );
};

