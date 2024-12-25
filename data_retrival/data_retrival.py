import requests as req
import pandas as pd


REPOS = ["vuejs/vue", "tensorflow/tensorflow", "twbs/bootstrap", "flutter/flutter", "angular/angular",
 "opencv/opencv", "kubernetes/kubernetes", "rust-lang/rust", "microsoft/TypeScript", "nodejs/node"]

print(f"Retriving from {len(REPOS)} repos")