from django.conf.urls import url,re_path
from kinase_predictor import views

urlpatterns = [
    url(r"homepage",views.homepage),
    url(r"submit",views.submit),
    url(r"result",views.result),
    url(r"help",views.help),
    url(r"contact",views.contact),
    url(r"trysth",views.trysth),
    url(r'^molecule/([0-9]{1,3})$', views.molecule)

]