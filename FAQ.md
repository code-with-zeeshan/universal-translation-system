# 📋 FAQ: Universal Translation System

## 🔥 **Core Differentiation Questions**

### **Q1: Why not just use M2M-100 or NLLB-200 quantized?**

**A:** Our system uses an innovative edge-cloud split architecture with a universal encoder (35MB base + 2-4MB vocabulary packs) and cloud decoder infrastructure. This results in a 40MB app with 90% quality of full models. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### **Q2: What makes your vocabulary pack system unique?**

**A:** Our vocabulary packs are small (2-4MB each), dynamically loaded, and language-specific. This allows users to download only the languages they need. See [vocabulary/Vocabulary_Guide.md](vocabulary/Vocabulary_Guide.md).

### **Q3: How do you maintain quality with such a small model?**

**A:** Through smart quantization, optimized vocabulary packs, and our edge-cloud split architecture. The heavy lifting is done on the cloud decoder while keeping the client-side encoder lightweight. See [docs/VISION.md](docs/VISION.md) for details.

---

## 💡 **Technical Architecture Questions**

### **Q4: Why split encoder and decoder?**

**A:** This split architecture minimizes client app size while maximizing translation quality. The encoder runs on the device, while the decoder runs in the cloud. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### **Q5: How is privacy preserved if you're using cloud?**

**A:** Only embeddings are sent to the cloud; the original text never leaves the device. These embeddings are compressed and cannot be reversed to obtain the original text.

### **Q6: What about offline translation?**

**A:** While our primary architecture is edge-cloud, we're working on a fully offline mode for scenarios where internet connectivity is limited or unavailable.

---

## 🚀 **Business/User Questions**

### **Q7: Who is this system designed for?**

**A:** Developers, privacy-conscious users, and organizations needing scalable, monitored translation. Our SDKs support Android, iOS, Flutter, React Native, and Web. See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md).

### **Q8: How does the configuration work?**

**A:** All components are configurable via environment variables, making deployment and customization easy. See [docs/environment-variables.md](docs/environment-variables.md).

### **Q9: What languages are supported?**

**A:** We currently support 20 languages with plans to expand. The coordinator dashboard shows all available languages and their status.

---

## 🔧 **Developer/Technical Integration Questions**

### **Q10: How hard is it to integrate into existing apps?**

**A:** Our SDKs are designed for easy integration with minimal code. See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for platform-specific examples.

### **Q11: Can I use my own vocabulary/terminology?**

**A:** Yes, you can create custom vocabulary packs for domain-specific terminology. See [vocabulary/Vocabulary_Guide.md](vocabulary/Vocabulary_Guide.md).

### **Q12: How does deployment work?**

**A:** We provide Docker and Kubernetes configurations for easy deployment. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## 🎯 **Future/Roadmap Questions**

### **Q13: What's the roadmap?**

**A:** We're focusing on improving offline capabilities, adding more languages, and enhancing the monitoring dashboard. See our [Future Roadmap](docs/future_plan.md) for details.

### **Q14: How do you handle model updates?**

**A:** Updates are managed through our environment variable configuration system, allowing for seamless updates without code changes.

### **Q15: What makes this system unique?**

**A:** Our edge-cloud split architecture, dynamic vocabulary system, and environment variable configuration make this system highly flexible, efficient, and privacy-focused.

---

For more information, see the documentation in the `/docs` folder and explore the coordinator dashboard for real-time system status.