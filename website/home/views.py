from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt,csrf_protect

from .forms import DateTimeRangeForm

@csrf_exempt
# Create your views here.
def index(request):
    index_type = 0
    if (request.method == 'POST'):
        form = DateTimeRangeForm(request.POST)
        if form.is_valid():
            index_type = 1
            date = form.cleaned_data['date']
            from_time = form.cleaned_data['from_time']
            to_time = form.cleaned_data['to_time']
            if (date.year == 2022 and date.month == 1 and date.day == 1 and from_time.hour == 3 and from_time.minute == 30 and to_time.hour == 8 and to_time.minute == 30):
                index_type = 2
    return render(request, "home/index.html", {'index_type': index_type})
