import { useState, useRef, useEffect } from 'react'
import { Camera, UploadCloud, AlertCircle, FileWarning } from 'lucide-react'

// Dummy prediction exactly matching the legacy TFJS frontend
const DUMMY_PREDICTION = [
    { className: 'Normal', probability: 0.982 },
    { className: 'Tuberculosis', probability: 0.018 }
]

function App() {
    const [selectedImage, setSelectedImage] = useState(null)
    const [imageFile, setImageFile] = useState(null)
    const [predictions, setPredictions] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const fileInputRef = useRef(null)

    // Load the default image on mount, mimicking the old app
    useEffect(() => {
        // We will place the default image in the public folder
        setSelectedImage('/assets/tb_image.jpg')
        // Simulate initial prediction
        setLoading(true)
        setTimeout(() => {
            setPredictions(DUMMY_PREDICTION)
            setLoading(false)
        }, 1500)
    }, [])

    const handleImageChange = (e) => {
        const file = e.target.files[0]
        if (file) {
            setImageFile(file)
            const reader = new FileReader()
            reader.onload = (e) => {
                setSelectedImage(e.target.result)
            }
            reader.readAsDataURL(file)

            // Clear previous states
            setPredictions(null)
            setError(null)

            // Attempt prediction
            makePrediction(file)
        }
    }

    const makePrediction = async (file) => {
        setLoading(true)
        setError(null)

        // Default to the dummy local prediction if backend fails
        // This allows the UI to be testable even if backend isn't running yet
        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`)
            }

            const data = await response.json()
            setPredictions(data.predictions)
        } catch (err) {
            console.warn("Backend API failed, falling back to dummy data for demonstration.", err)
            setError("Backend API not reachable. Showing dummy data.")

            // Fallback
            setTimeout(() => {
                setPredictions(DUMMY_PREDICTION)
                setLoading(false)
            }, 1000)
        } finally {
            // If we didn't hit the catch block's timeout
            if (!error) setLoading(false)
        }
    }

    return (
        <div className="app-container">
            <header className="header">
                <h1>Tuberculosis Analyzer</h1>
                <p>AI powered web app to diagnose TB from chest x-rays.</p>
            </header>

            <main className="main-card">

                {/* Image Display */}
                <div className="image-container">
                    {selectedImage ? (
                        <img
                            src={selectedImage}
                            alt="Chest X-Ray"
                            className="uploaded-image"
                        />
                    ) : (
                        <div className="placeholder-content">
                            <UploadCloud size={48} />
                            <p>No image selected</p>
                        </div>
                    )}
                </div>

                {/* Action Button */}
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageChange}
                    accept="image/jpeg, image/png"
                    className="hidden-input"
                />

                <button
                    className="upload-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading}
                >
                    <Camera size={20} />
                    {loading ? 'Analyzing...' : 'Analyze Image'}
                </button>

                {/* Loading Indicator */}
                {loading && (
                    <div className="loading-container">
                        <UploadCloud className="spinner" size={24} />
                        <span>Model is predicting...</span>
                    </div>
                )}

                {/* Error / Warning */}
                {error && !loading && (
                    <div style={{ marginTop: '1rem', color: '#ea580c', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AlertCircle size={18} />
                        <span>{error}</span>
                    </div>
                )}

                {/* Results */}
                {predictions && !loading && (
                    <div className="results-section">
                        <div className="results-header">
                            <FileWarning size={18} />
                            <span>Results</span>
                        </div>

                        <div className="prediction-list">
                            {predictions.map((p, idx) => (
                                <div key={idx} className="prediction-item">
                                    <span className="class-name">{p.className}</span>
                                    <span className="probability">{(p.probability).toFixed(3)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

            </main>

            <footer className="footer">
                <p>
                    Powered by React & FastAPI — <a href="https://woza.work/" target="_blank" rel="noreferrer">woza.work</a>
                </p>
            </footer>
        </div>
    )
}

export default App
