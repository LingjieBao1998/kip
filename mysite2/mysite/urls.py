"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,re_path
# import hello.views
# import kinase_predictor.views
from django.conf.urls import url, include
import kinase_predictor.views as views
from django.views import static ##新增
from django.conf import settings ##新增


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r"submit",views.submit),
    url(r"result",views.result),
    url(r"help",views.help),
    url(r"contact",views.contact),
    # url(r"trysth",views.trysth),
    # url(r'^molecule/([0-9]{1,3})$', views.molecule),
    url(r"molecule",views.molecule),
    url(r'^$',views.homepage),
    url(r"download",views.download,name="download"),
    # path("index/", login.views.index),#4. 编写路由
    # path("personinfo/", hello.views.personinfo,name="personinfo"),
    # url(r"^kinase_predictor/", include("kinase_predictor.urls")),
    ##　以下是新增
    url(r'^/static/(?P<path>.*)$', static.serve,
      {'document_root': settings.STATIC_ROOT}, name='static'),
]
