# Detection of Manipulated and Authentic Images

---

## Problem Definition

Distinguish authentic from digitally manipulated images using neural networks.

---

## Project Goal

Develop custom CNN from scratch without pre-trained models.

---

## Dataset

57,589 Kaggle images split into train, validation, and test sets.

---

## Data Preparation

Resize to 256x256, normalize values, and apply data augmentation.

---

## Model Architecture

Four convolutional blocks with batch normalization, max pooling, and dropout.

---

## Training Configuration

Adam optimizer, binary crossentropy loss, 20 epochs, and early stopping.

---

## Evaluation Metrics

Assessed using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## Key Results

Model achieved approximately 75.76% accuracy on unseen test data.

---

## Future Work

Explore advanced augmentation, new architectures, and explainable AI techniques.

---

## VBA Code to Generate Slides

```vba
Sub CreatePresentation()
    Dim pptApp As Object
    Dim pptPres As Object
    Dim pptSlide As Object
    
    ' Create instance of PowerPoint
    On Error Resume Next
    Set pptApp = GetObject(, "PowerPoint.Application")
    If Err.Number <> 0 Then
        Set pptApp = CreateObject("PowerPoint.Application")
    End If
    On Error GoTo 0
    
    ' Make it visible
    pptApp.Visible = True
    
    ' Create a new presentation
    Set pptPres = pptApp.Presentations.Add
    
    ' --- Slide 1 ---
    Set pptSlide = pptPres.Slides.Add(1, 1) ' 1 = ppLayoutTitle
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Detection of Manipulated and Authentic Images"
    
    ' --- Slide 2 ---
    Set pptSlide = pptPres.Slides.Add(2, 2) ' 2 = ppLayoutText
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Problem Definition"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Distinguish authentic from digitally manipulated images using neural networks."
    
    ' --- Slide 3 ---
    Set pptSlide = pptPres.Slides.Add(3, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Project Goal"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Develop custom CNN from scratch without pre-trained models."
    
    ' --- Slide 4 ---
    Set pptSlide = pptPres.Slides.Add(4, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Dataset"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "57,589 Kaggle images split into train, validation, and test sets."
    
    ' --- Slide 5 ---
    Set pptSlide = pptPres.Slides.Add(5, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Data Preparation"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Resize to 256x256, normalize values, and apply data augmentation."
    
    ' --- Slide 6 ---
    Set pptSlide = pptPres.Slides.Add(6, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Model Architecture"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Four convolutional blocks with batch normalization, max pooling, and dropout."
    
    ' --- Slide 7 ---
    Set pptSlide = pptPres.Slides.Add(7, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Training Configuration"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Adam optimizer, binary crossentropy loss, 20 epochs, and early stopping."
    
    ' --- Slide 8 ---
    Set pptSlide = pptPres.Slides.Add(8, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Evaluation Metrics"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Assessed using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix."
    
    ' --- Slide 9 ---
    Set pptSlide = pptPres.Slides.Add(9, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Key Results"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Model achieved approximately 75.76% accuracy on unseen test data."
    
    ' --- Slide 10 ---
    Set pptSlide = pptPres.Slides.Add(10, 2)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Future Work"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Explore advanced augmentation, new architectures, and explainable AI techniques."
    
    MsgBox "Presentation created successfully!", vbInformation
End Sub
```
