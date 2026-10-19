/**
 * main.jsx
 *
 * React application entry point.
 *
 * Renders the App component into the DOM root element with StrictMode enabled.
 * StrictMode activates additional development-time checks:
 * - Double-invocation of effects (to catch side-effect bugs)
 * - Deprecation warnings for legacy APIs
 * - Strict context validation
 *
 * These checks are automatically disabled in production builds.
 */

import {StrictMode} from "react";
import {createRoot} from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(<StrictMode>
    <App/>
</StrictMode>);
