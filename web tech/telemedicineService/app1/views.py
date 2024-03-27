from django.contrib import messages
from django.contrib.auth.models import User, auth
from django.shortcuts import HttpResponseRedirect, redirect, render
from app1.models import PatientRegistration, DoctorRegistration, PatientRecords

# Create your views here.

def home(request):
    obj = PatientRecords.objects.all()
    print(obj)
    print(obj[0].name)
    return render(request=request, template_name="index.html")

def signup(request):
    if request.method == 'POST':
        if 'submit_patient' in request.POST:
            username = request.POST.get('patient_name')
            patientEmail = request.POST.get('patient_email')
            phone = request.POST.get('patient_phone')
            dob = request.POST.get('patient_dob')
            gender = request.POST.get('patient_gender')
            patient_password = request.POST.get('patient_password')
            patient_password2 = request.POST.get('patient_password2')

            if User.objects.filter(email=patientEmail).exists():
                messages.info(request, 'Email already taken')
                print("Invalid credentials")
                return redirect('signup_page')
            
            if patient_password != patient_password2:
                print("Password not match")
                messages.info(request, 'Passwords do not match')
            else:
                user = User.objects.create_user(username=username, email=patientEmail, password=patient_password)
                model_user = PatientRegistration(user=user, phone_number=phone, gender=gender, date_of_birth=dob)
                user.save()
                model_user.save()
                messages.success(request, "User Created Successfully")
                print("user created")
                return redirect('login_page')


        elif 'submit_doctor' in request.POST:
            username = request.POST.get('doctor_name')
            doctorEmail = request.POST.get('doctor_email')
            doctor_phone = request.POST.get('doctor_phone')
            doctor_password = request.POST.get('doctor_password')
            doctor_password2 = request.POST.get('doctor_password2')
            doctor_speciality = request.POST.get('doctor_speciality')

            if User.objects.filter(email=doctorEmail).exists():
                messages.info(request, 'Email already taken')
                print("Invalid credentials")
                return redirect('signup_page')
            
            if doctor_password != doctor_password2:
                print("Password not match")
                messages.info(request, 'Passwords do not match')
            else:
                doctor_user = User.objects.create_user(username=username, email=doctorEmail, password=doctor_password)
                model_user2 = DoctorRegistration(doctor_user=doctor_user, phone_number=doctor_phone, doctor_speciality=doctor_speciality)
                doctor_user.save()
                model_user2.save()
                messages.success(request, "User Created Successfully")
                print("user created")
                return redirect('login_page')

    return render(request=request, template_name="signup.html")

def login(request):
    if not request.user.is_authenticated:
        if request.method == 'POST':
            if 'login_patient' in request.POST:
                username = request.POST.get('patient_name')
                password = request.POST.get('patient_password')
                user = auth.authenticate(username=username, password=password)
                print(username)
                print(password)
                if user is not None:
                    auth.login(request, user)
                    return redirect('service_page')
                else:
                    messages.info(request, 'Invalid credentials')
                    print("Invalid credentials")
                    return redirect('login_page')
                
            elif 'login_doctor' in request.POST:
                username = request.POST.get('doctor_name')
                password = request.POST.get('doctor_password')
                user2 = auth.authenticate(username=username, password=password)
                print(username)
                print(password)
                if user2 is not None:
                    auth.login(request, user2)
                    return redirect('img_process_page')
                else:
                    messages.info(request, 'Invalid credentials')
                    print("Invalid credentials")
                    return redirect('login_page')
        return render(request=request, template_name="login.html")
    else:
        return redirect('home_page')

def logout(request):
    auth.logout(request)
    return redirect('home_page')


def service(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            patient_registration = PatientRegistration.objects.get(user=request.user)
            name = request.POST.get('name')
            age = request.POST.get('age')
            gender = request.POST.get('gender')
            mobile = request.POST.get('mobile')
            bloodpressure = request.POST.get('bloodpressure')
            heartrate = request.POST.get('heartrate')
            stresslevel = request.POST.get('stresslevel')
            diabetes = request.POST.get('diabetes')
            chestDiseaseDuration = request.POST.get('chestDiseaseDuration')
            healthIssues = request.POST.get('healthIssues')
            lastCheckupDate = request.POST.get('lastCheckupDate')
            previousDoctorAdvice = request.POST.get('previousDoctorAdvice')
            recentCondition = request.POST.get('recentCondition')
            xray_img = request.FILES.get('image')

            user_records = PatientRecords(patient_registration=patient_registration, name=name, age=age, gender=gender, phone_number=mobile, blood_pressure=bloodpressure, heart_rate=heartrate, stress_level=stresslevel, diabetes=diabetes, chestDiseaseDuration=chestDiseaseDuration, healthIssues=healthIssues,lastCheckupDate=lastCheckupDate, previousDoctorAdvice=previousDoctorAdvice, recentCondition=recentCondition, xray_img=xray_img)

            user_records.save()
            

            return redirect('home_page')

        return render(request=request, template_name="services.html")
    else:
        return redirect('login_page')

def imageProcessing(request):
    if request.user.is_authenticated:
        return render(request=request, template_name="imageprocessing.html")
    else:
        return redirect('login_page')
