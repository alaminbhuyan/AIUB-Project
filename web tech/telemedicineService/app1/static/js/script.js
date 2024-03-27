// const wrapper = document.querySelector('.wrapper');
// const loginlink = document.querySelector('.login-link');
// const registerlink = document.querySelector('.register-link');
// const btnPopup = document.querySelector('.btnLogin-popup'); 
// const btnPopupPaitent = document.querySelector('.btnLogin-popup-patient'); 
// const iconClose = document.querySelector('.icon-close');

// const btnSubmitLog = document.getElementById("btn-submit-log");
// const btnSubmitReg = document.getElementById("btn-submit-reg");

// registerlink.addEventListener('click', ()=> {
//     wrapper.classList.add('active');
// });

// loginlink.addEventListener('click', ()=> {
//     wrapper.classList.remove('active');
// });

// btnPopup.addEventListener('click', ()=> {
//     wrapper.classList.add('active-popup');
//     wrapper.classList.remove('active');

//     docPatLog.classList.remove('active');
//     btnSubmitLog.textContent = 'Login as Doctor';
// });

// btnPopupPaitent.addEventListener('click', ()=> {
//   wrapper.classList.add('active-popup');
//   wrapper.classList.remove('active');

//   docPatLog.classList.add('active');
//   btnSubmitLog.textContent = 'Login as Patient';

// });

// iconClose.addEventListener('click', ()=> {
//     wrapper.classList.remove('active-popup');
//     wrapper.classList.remove('active');

//     //for doc pat toggle
//     docPatLog.classList.remove('active');
//     docPatReg.classList.remove('active');

// });

// doctor and patient toggle
// const docPatLog = document.querySelector('.doc-pat-log');
// const docPatReg = document.querySelector('.doc-pat-reg');
// const btnDocLog = document.getElementById("btn-doc-log");
// const btnPatLog = document.getElementById("btn-pat-log");
// const btnDocReg = document.getElementById("btn-doc-reg");
// const btnPatReg = document.getElementById("btn-pat-reg");

// btnDocLog.addEventListener('click', ()=> {
//   docPatLog.classList.remove('active');
//   btnSubmitLog.textContent = 'Login as Doctor';
// });

// btnPatLog.addEventListener('click', ()=> {
//   docPatLog.classList.add('active');
//   btnSubmitLog.textContent = 'Login as Patient';
// });

// btnDocReg.addEventListener('click', ()=> {
//   docPatReg.classList.remove('active');
//   btnSubmitReg.textContent = 'Register as Doctor';
// });

// btnPatReg.addEventListener('click', ()=> {
//   docPatReg.classList.add('active');
//   btnSubmitReg.textContent = 'Register as Patient';
// });

// BMI start //

var age = document.getElementById("age");
var height = document.getElementById("height");
var weight = document.getElementById("weight");
var male = document.getElementById("m");
var female = document.getElementById("f");
var form = document.getElementById("form");
var rslt = document.getElementById("rslt");
var h1, h2;

function validateForm(){
  if(age.value=='' || height.value=='' || weight.value=='' || (male.checked==false && female.checked==false)){
    alert("All fields are required!");
    //document.getElementById("submit").removeEventListener("click", countBmi);
  }else{
    if(rslt.contains(h1) && rslt.contains(h2)){
    h1.remove();
    h2.remove();
    }
    countBmi();
  }
}
document.getElementById("submit").addEventListener("click", validateForm);

function countBmi(){
  var p = [age.value, height.value, weight.value];
  if(male.checked){
    p.push("male");
  }else if(female.checked){
    p.push("female");
  }
  form.reset();
  var bmi = Number(p[2])/(Number(p[1])/100*Number(p[1])/100);
      
  var result = '';
  if(bmi<18.5){
    result = 'Underweight';
     }else if(18.5<=bmi&&bmi<=24.9){
    result = 'Healthy';
     }else if(25<=bmi&&bmi<=29.9){
    result = 'Overweight';
     }else if(30<=bmi&&bmi<=34.9){
    result = 'Obese';
     }else if(35<=bmi){
    result = 'Extremely obese';
     }
  
    h1 = document.createElement("h1");
    h2 = document.createElement("h2");

  var t = document.createTextNode(result);
  var b = document.createTextNode('BMI: ');
  var r = document.createTextNode(parseFloat(bmi).toFixed(2));

  h1.appendChild(t);
  h2.appendChild(b);
  h2.appendChild(r);
  
  //document.body.appendChild(h1);
  //document.body.appendChild(h2);

  rslt.appendChild(h1);
  rslt.appendChild(h2);
  
  //document.getElementById("submit").removeEventListener("click", countBmi);
  //document.getElementById("submit").removeEventListener("click", validateForm);
}
//document.getElementById("submit").addEventListener("click", countBmi);

// BMI start //