//
//  ImageViewController.swift
//  TwoMLModels
//
//  Created by Olivia Bishop on 10/31/19.
//  Copyright © 2019 Olivia Bishop. All rights reserved.
//
// ImageViewController.swift
//  ML-Vision
//
//  Created by DALE MUSSER on 12/12/17.
//  Updated 10/26/18 for Xcode 10.0
//  Copyright © 2017 Tech Innovator. All rights reserved.
//
// http://www.wolfib.com/Image-Recognition-Intro-Part-1/
// https://developer.apple.com/machine-learning///



import UIKit
import CoreML
import Vision

class ImageViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var googleTextView: UITextView!
    
    
     let imagePicker = UIImagePickerController()
    
    
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        imagePicker.delegate = self
        textView.text = ""
        googleTextView.text = ""
        activityIndicator.hidesWhenStopped = true
        // Do any additional setup after loading the view.
    }
    

    @IBAction func cameraSelected(_ sender: Any) {

        takePhotoWithCamera()
    }
    
    @IBAction func photoLibrarySelected(_ sender: Any) {
        
        pickPhotoFromLibrary()
    }
    
    func takePhotoWithCamera() {
        if (!UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera)) {
            let alertController = UIAlertController(title: "No Camera", message: "The device has no camera.", preferredStyle: .alert)
            let okAction = UIAlertAction(title: "OK", style: .default, handler: nil)
            alertController.addAction(okAction)
            present(alertController, animated: true, completion: nil)
        } else {
            imagePicker.allowsEditing = false
            imagePicker.sourceType = .camera
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    
     func pickPhotoFromLibrary() {
           imagePicker.allowsEditing = false
           imagePicker.sourceType = .photoLibrary
           present(imagePicker, animated: true, completion: nil)
       }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
            textView.text = ""
            
            guard let ciImage = CIImage(image: pickedImage) else {
                displayString(string: "Unable to convert image to CIImage.");
                return
            }
            
            detectScene(image: ciImage)
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func displayString(string: String) {
        textView.text = textView.text + string + "\n";
    }
    
    func displayGoogleString(string: String) {
        googleTextView.text = googleTextView.text + string + "\n";
    }
    
    func detectScene(image: CIImage) {
        displayString(string: "detecting scene...")
        
        // Load the ML model through its generated class
        
        guard let model = try? VNCoreMLModel(for: VGG16().model) else {
            displayString(string: "Can't load ML model.")
            return
        }
        
        
        // Create a Vision request with completion handler
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let _ = results.first else {
                    self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                for result in results {
                    self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                }
            }
        }
        
        activityIndicator.startAnimating()
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayString(string: error.localizedDescription)
                    self?.activityIndicator.stopAnimating()
                }
            }
        }
    }
    
     func detectGoogleScene(image: CIImage) {
               displayGoogleString(string: "detecting Google scene...")
               
               guard let model = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
               displayString(string: "Can't load ML model.")
               return
           }
        
         let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                   guard let results = request.results as? [VNClassificationObservation],
                       let _ = results.first else {
                           self?.displayGoogleString(string: "Unexpected result type from VNCoreMLRequest")
                           return
                   }
            
        DispatchQueue.main.async { [weak self] in
        self?.activityIndicator.stopAnimating()
        for result in results {
            self?.displayGoogleString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
        }
    }
}
         activityIndicator.startAnimating()
        
     let handler = VNImageRequestHandler(ciImage: image)
           DispatchQueue.global(qos: .userInteractive).async {
               do {
                   try handler.perform([request])
               } catch {
                   DispatchQueue.main.async { [weak self] in
                       self?.displayGoogleString(string: error.localizedDescription)
                       self?.activityIndicator.stopAnimating()
                }
            }
        }
}
}
