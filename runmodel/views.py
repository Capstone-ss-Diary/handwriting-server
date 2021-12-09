from django.shortcuts import render
from .models import HandWriting
import sys
from django.views.decorators.csrf import csrf_exempt

sys.path.insert(
    1, 'runmodel/')
import handwriting_function

@csrf_exempt
def handwriting(request):

    if request.method == "POST":
      hand_writing = HandWriting()
      hand_writing.user_id = request.session.get("user")
      hand_writing.image = request.FILES.get("chooseFile")
      hand_writing.save()

      data_file = HandWriting.objects.filter(user_id=request.session.get("user"))

      handwriting_function.create_handwriting_dataset(data_file[len(data_file)-1].image)
      handwriting_function.train_handwriting()
      handwriting_function.infer_handwriting()
      handwriting_function.create_svg_files()
      handwriting_function.download_fontello()



    return render(request, "runmodel/handwriting.html")

def example(request):
  return render(request, "runmodel/example.html")